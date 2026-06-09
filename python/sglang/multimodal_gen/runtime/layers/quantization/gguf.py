from __future__ import annotations

# GGUF quantization support for diffusion transformers (DiT).
#
# GGUF is the dominant distribution format for community quantized diffusion
# checkpoints (Flux / SD3 / Wan, e.g. the city96 exports). This adapter keeps
# the weights packed (uint8) in VRAM and dequantizes on-the-fly during the
# linear forward, which is what lets a large DiT fit on a small GPU.
#
# Unlike the LLM (`srt`) GGUF path, diffusion activations are N-D
# (`[batch, seq, hidden]`) rather than 2D token streams, so the fused
# `ggml_mul_mat_*` kernels (which expect 2D inputs) do not apply. We therefore
# use the dequant+GEMM path, exactly like vLLM-Omni's `DiffusionGGUFLinearMethod`.
# The dequant kernel and the shared GGUF constants/parameter type are reused
# from `srt` so there is a single source of truth for the on-the-fly dequant.

import logging
from typing import Any

import torch
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.utils import set_weight_attrs

logger = logging.getLogger(__name__)


def _dequantize_gguf(
    qweight: torch.Tensor, qweight_type: int, out_dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize a single GGUF weight tensor to ``out_dtype``.

    Leans on the `srt` GGUF kernel binding (`ggml_dequantize`) and the gguf
    block-size table so there is exactly one dequant implementation in the tree.
    """
    import gguf

    from sglang.srt.layers.quantization.gguf import UNQUANTIZED_TYPES

    if qweight_type in UNQUANTIZED_TYPES:
        return qweight.to(out_dtype)

    # Lazy import: the kernel is only built/available on CUDA/MUSA. Importing it
    # lazily keeps `gguf.py` importable (e.g. for registration) on other
    # platforms and surfaces a clear error only when a GGUF weight is actually
    # used.
    try:
        from sgl_kernel.quantization import ggml_dequantize
    except ImportError as err:  # pragma: no cover - depends on the build
        raise RuntimeError(
            "GGUF dequantization requires the sgl_kernel CUDA/MUSA build "
            "(sgl_kernel.quantization.ggml_dequantize)."
        ) from err

    block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
    rows = qweight.shape[0]
    cols = qweight.shape[1] // type_size * block_size
    return ggml_dequantize(qweight, qweight_type, rows, cols, out_dtype)


def dequant_gemm_gguf(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    """``x @ dequant(qweight).T`` for N-D diffusion activations."""
    weight = _dequantize_gguf(qweight, qweight_type, x.dtype)
    return x @ weight.T


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF-quantized diffusion transformers."""

    def __init__(self, modules_to_not_convert: list[str] | None = None) -> None:
        super().__init__()
        self.modules_to_not_convert = modules_to_not_convert or []

    def __repr__(self) -> str:
        return f"GGUFConfig(modules_to_not_convert={self.modules_to_not_convert})"

    @classmethod
    def get_name(cls) -> str:
        return "gguf"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GGUFConfig":
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            return GGUFLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []


def is_layer_skipped_gguf(prefix: str, modules_to_not_convert: list[str]) -> bool:
    return any(module_name in prefix for module_name in modules_to_not_convert)


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF using dequant+GEMM (N-D safe).

    Weights are stored packed as ``qweight`` (uint8) plus a per-shard
    ``qweight_type`` and dequantized on each forward pass.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.quantization.gguf import GGUFUninitializedParameter

        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        tensor_shape = (output_size_per_partition, input_size_per_partition)

        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        # GGUF params accumulate quantized shards in a data_container, so they
        # need a dedicated weight loader rather than the layer's dense one.
        # Override directly: extra_weight_attrs already set `weight_loader`, and
        # set_weight_attrs refuses to overwrite an existing attribute.
        qweight.weight_loader = self._gguf_weight_loader
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight_type,
            {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        qweight_type.weight_loader = self._gguf_weight_loader
        layer.register_parameter("qweight_type", qweight_type)

    @staticmethod
    def _gguf_weight_loader(
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str | int | None = None,
    ) -> None:
        """Load a GGUF ``qweight`` / ``qweight_type`` tensor.

        For unmerged layers ``shard_id`` is None and the weight is materialized
        directly. For merged layers (e.g. fused QKV) each shard arrives
        separately and is accumulated in ``data_container`` keyed by
        ``shard_id``; ``process_weights_after_loading`` later pads and concats
        them. Mirrors the `srt`/vLLM GGUF loader, kept self-contained on the
        parameter so it does not depend on the dense linear weight loaders.
        """
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        is_gguf_weight = getattr(param, "is_gguf_weight", False)

        if is_gguf_weight_type:
            weight_type = loaded_weight.item()
            if shard_id is not None:
                param.shard_weight_type[shard_id] = weight_type
            param.weight_type = weight_type
            return

        if not is_gguf_weight:
            param.data.copy_(loaded_weight)
            return

        if shard_id is not None:
            # Merged layer: stash the shard, materialize on post-load.
            param.shard_id.append(shard_id)
            param.shard_id_map[shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            return

        # Unmerged layer: materialize the uninitialized parameter in place.
        from torch.nn.parameter import UninitializedParameter

        if isinstance(param, UninitializedParameter):
            param.materialize(
                tuple(loaded_weight.shape),
                device=loaded_weight.device,
                dtype=loaded_weight.dtype,
            )
        param.data.copy_(loaded_weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from sglang.srt.layers.quantization.gguf import (
            DEQUANT_TYPES,
            UNQUANTIZED_TYPES,
        )
        from gguf import GGMLQuantizationType as WeightType

        qweight_type = layer.qweight_type.weight_type
        # For merged layers the top-level weight_type may be unset (0); the
        # per-shard types are validated implicitly when dequantized.
        if layer.qweight_type.shard_weight_type:
            types_to_check = layer.qweight_type.shard_weight_type.values()
        else:
            types_to_check = [qweight_type]
        for wtype in types_to_check:
            if wtype not in UNQUANTIZED_TYPES and wtype not in DEQUANT_TYPES:
                raise ValueError(
                    f"Unsupported GGUF quantization type {WeightType(wtype)} "
                    f"in layer {layer}."
                )
        self._create_padded_weight_param(layer)

    def _create_padded_weight_param(self, layer: torch.nn.Module) -> None:
        """Pad+concat the per-shard quantized weights of a merged layer.

        Different shards (q/k/v) may use different quant types and therefore
        different packed widths, so we pad to the max width and record where
        each shard lives via ``shard_offset_map``. Single-shard layers are left
        as-is. Ported from `srt`'s GGUFLinearMethod.
        """
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1, f"Data container has mixed dtypes: {dtype}"
            dtype = next(iter(dtype))
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            padded_data = torch.zeros(
                (concat_side, padded_side),
                dtype=dtype,
                device=data_container[0].device,
            )
            # (dim0_start, dim0_end, dim1_size)
            shard_offset_map: dict[Any, tuple[int, int, int]] = {}
            for idx in shard_id:
                id_in_container = shard_id_map[idx]
                start = sum(x.size(0) for x in data_container[:id_in_container])
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
            qweight.data_container.clear()
            padded_param = Parameter(padded_data, requires_grad=False)
            set_weight_attrs(padded_param, vars(qweight))
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            layer.register_parameter("qweight", padded_param)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shard_offset_map = getattr(layer.qweight, "shard_offset_map", None)

        if shard_offset_map:
            # Merged layer: dequantize each shard with its own quant type, then
            # concat along the output (feature) dimension. Iterate in shard-id
            # order (0,1,2 -> q,k,v / gate,up) so the fused output layout matches
            # what the dense merged layer would have produced.
            qweight = layer.qweight
            result = []
            for idx in sorted(shard_offset_map.keys()):
                start, end, offset = shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type[idx]
                result.append(
                    dequant_gemm_gguf(
                        x, qweight[start:end, :offset].contiguous(), qweight_type
                    )
                )
            out = torch.cat(result, axis=-1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = dequant_gemm_gguf(x, qweight, qweight_type)

        if bias is not None:
            out.add_(bias)
        return out
