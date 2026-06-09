# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter, UninitializedParameter

from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.models.utils import WeightsMapper


class LiteLinearConfig(QuantizationConfig):
    """Optional LiteLinear integration for dense FFN projections."""

    DEFAULT_MIN_INPUT_SIZE = 4096
    DEFAULT_MIN_OUTPUT_RATIO = 2.0
    DEFAULT_MAX_OUTPUT_RATIO = 12.0
    MODEL_LOADER_EXTRA_CONFIG_KEYS = [
        "litelinear_checkpoint_mode",
        "litelinear_online_patch",
        "litelinear_strict_patch",
        "litelinear_patch_cache_root",
        "litelinear_patch_config_path",
        "litelinear_patch_tag",
        "litelinear_patch_copy_original",
        "litelinear_patch_force_rebuild",
        "litelinear_rank",
    ]

    def __init__(
        self,
        rank: int = 64,
        target_patterns: Optional[List[str]] = None,
        ignored_layers: Optional[List[str]] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
        min_input_size: int = DEFAULT_MIN_INPUT_SIZE,
        min_output_ratio: float = DEFAULT_MIN_OUTPUT_RATIO,
        max_output_ratio: float = DEFAULT_MAX_OUTPUT_RATIO,
        load_dense_weight: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LiteLinear rank must be positive, got {rank}.")
        if min_input_size <= 0:
            raise ValueError(
                f"LiteLinear min_input_size must be positive, got {min_input_size}."
            )
        if min_output_ratio <= 0:
            raise ValueError(
                "LiteLinear min_output_ratio must be positive, "
                f"got {min_output_ratio}."
            )
        if max_output_ratio < min_output_ratio:
            raise ValueError(
                "LiteLinear max_output_ratio must be greater than or equal to "
                f"min_output_ratio, got {max_output_ratio} < {min_output_ratio}."
            )

        self.rank = rank
        self.load_dense_weight = load_dense_weight
        self.min_input_size = min_input_size
        self.min_output_ratio = min_output_ratio
        self.max_output_ratio = max_output_ratio
        self.target_patterns = self._normalize_string_list(
            target_patterns, default=[]
        )
        self.ignored_layers = self._normalize_string_list(ignored_layers, default=[])
        self.packed_modules_mapping = packed_modules_mapping or {}
        self._compiled_target_patterns = [
            re.compile(pattern) for pattern in self.target_patterns
        ]

    @classmethod
    def get_name(cls) -> str:
        return "litelinear"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LiteLinearConfig":
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert"], None
        )
        target_patterns = cls.get_from_keys_or(
            config, ["target_patterns", "litelinear_target_patterns"], None
        )
        return cls(
            rank=int(cls.get_from_keys_or(config, ["rank", "litelinear_rank"], 64)),
            target_patterns=target_patterns,
            ignored_layers=ignored_layers,
            packed_modules_mapping=cls.get_from_keys_or(
                config, ["packed_modules_mapping"], {}
            ),
            min_input_size=int(
                cls.get_from_keys_or(
                    config,
                    ["min_input_size", "litelinear_min_input_size"],
                    cls.DEFAULT_MIN_INPUT_SIZE,
                )
            ),
            min_output_ratio=float(
                cls.get_from_keys_or(
                    config,
                    ["min_output_ratio", "litelinear_min_output_ratio"],
                    cls.DEFAULT_MIN_OUTPUT_RATIO,
                )
            ),
            max_output_ratio=float(
                cls.get_from_keys_or(
                    config,
                    ["max_output_ratio", "litelinear_max_output_ratio"],
                    cls.DEFAULT_MAX_OUTPUT_RATIO,
                )
            ),
            load_dense_weight=bool(
                cls.get_from_keys_or(
                    config,
                    ["load_dense_weight", "litelinear_load_dense_weight"],
                    True,
                )
            ),
        )

    @classmethod
    def get_model_loader_extra_config_keys(cls) -> List[str]:
        return cls.MODEL_LOADER_EXTRA_CONFIG_KEYS

    def update_from_model_loader_extra_config(
        self, model_loader_extra_config: Dict[str, Any]
    ) -> None:
        checkpoint_mode = _get_checkpoint_mode(model_loader_extra_config)
        if checkpoint_mode in ("online_patch", "strict"):
            self.load_dense_weight = False
            if "litelinear_rank" in model_loader_extra_config:
                self.rank = int(model_loader_extra_config["litelinear_rank"])

    @classmethod
    def maybe_process_safetensors_files(
        cls,
        hf_weights_files: List[str],
        model_loader_extra_config: Dict[str, Any],
    ) -> List[str]:
        return maybe_resolve_litelinear_patched_weights(
            hf_weights_files,
            model_loader_extra_config,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase

        if not isinstance(layer, LinearBase):
            return None

        if prefix in self.ignored_layers or is_layer_skipped(
            prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
        ):
            return UnquantizedLinearMethod()

        if not self._is_allowed_by_name(prefix) or not self._is_supported_shape(layer):
            return UnquantizedLinearMethod()

        return LiteLinearMethod(self, prefix)

    def get_scaled_act_names(self) -> List[str]:
        return []

    def apply_weight_name_mapper(self, hf_to_sglang_mapper: "WeightsMapper"):
        if self.ignored_layers:
            self.ignored_layers = list(
                dict.fromkeys(hf_to_sglang_mapper.apply_list(self.ignored_layers))
            )

    def _is_allowed_by_name(self, prefix: str) -> bool:
        if not self._compiled_target_patterns:
            return True
        if not prefix:
            return False
        return any(pattern.search(prefix) for pattern in self._compiled_target_patterns)

    def _is_supported_shape(self, layer: torch.nn.Module) -> bool:
        input_size = int(getattr(layer, "input_size", 0))
        output_size = int(getattr(layer, "output_size", 0))
        if min(input_size, output_size) < self.min_input_size:
            return False

        feature_ratio = max(output_size / input_size, input_size / output_size)
        return self.min_output_ratio <= feature_ratio <= self.max_output_ratio

    @staticmethod
    def _normalize_string_list(
        value: Optional[str | List[str]],
        default: List[str],
    ) -> List[str]:
        if value is None:
            return default
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return list(value)


class LiteLinearMethod(LinearMethodBase):
    FACTOR_NAMES = ("A", "B", "Q_fp8", "Q_scale_inv")

    def __init__(self, quant_config: LiteLinearConfig, prefix: str = "") -> None:
        self.quant_config = quant_config
        self.prefix = prefix

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        if self.quant_config.load_dense_weight:
            weight = Parameter(
                torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
            layer.register_parameter("weight", weight)
            set_weight_attrs(weight, extra_weight_attrs)
        else:
            layer.register_parameter("weight", None)

        self._register_lazy_factor_parameter(
            layer,
            "A",
            (output_size_per_partition, self.quant_config.rank),
            params_dtype,
        )
        self._register_lazy_factor_parameter(
            layer,
            "B",
            (self.quant_config.rank, input_size_per_partition),
            params_dtype,
        )
        self._register_lazy_factor_parameter(
            layer,
            "Q_fp8",
            (output_size_per_partition, input_size_per_partition),
            torch.float8_e4m3fn,
        )
        self._register_lazy_factor_parameter(layer, "Q_scale_inv", (), torch.float32)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_lite_linear_module", None) is not None:
            return

        if self._has_loaded_factors(layer):
            self._process_loaded_factors(layer)
            return

        if layer.weight is None:
            missing = ", ".join(
                name
                for name in self.FACTOR_NAMES
                if not getattr(getattr(layer, name, None), "_litelinear_loaded", False)
            )
            raise RuntimeError(
                f"LiteLinear layer {self.prefix or '<unnamed>'} was configured "
                "without dense weight fallback, but factor tensors were not "
                f"loaded. Missing: {missing}."
            )

        LiteLinear = _load_litelinear_class()
        weight = layer.weight
        lite_layer = _make_lite_linear(
            LiteLinear,
            in_features=weight.shape[1],
            out_features=weight.shape[0],
            rank=self.quant_config.rank,
            device=weight.device,
            dtype=weight.dtype,
        )

        with torch.no_grad():
            lite_layer.weight.copy_(weight)
            lite_bias = getattr(lite_layer, "bias", None)
            if lite_bias is not None:
                lite_bias.zero_()

        if self.prefix:
            setattr(lite_layer, "_lite_key", self.prefix)

        materialize_from_weight = getattr(lite_layer, "materialize_from_weight", None)
        if materialize_from_weight is None:
            raise RuntimeError(
                "lite_linear.LiteLinear does not expose materialize_from_weight()."
            )
        materialize_from_weight()
        lite_layer.eval()
        layer.register_parameter("weight", None)
        self._clear_factor_parameters(layer)

        # Avoid registering this temporary wrapper as a child module while
        # model.named_modules() is iterating during post-load processing.
        object.__setattr__(layer, "_lite_linear_module", lite_layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        lite_layer = getattr(layer, "_lite_linear_module", None)
        if lite_layer is None:
            self.process_weights_after_loading(layer)
            lite_layer = layer._lite_linear_module

        output = lite_layer(x)
        if output.dtype != x.dtype:
            output = output.to(dtype=x.dtype)
        if bias is not None:
            output = output + bias
        return output

    def _register_lazy_factor_parameter(
        self,
        layer: torch.nn.Module,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        param = UninitializedParameter(requires_grad=False)
        setattr(param, "_litelinear_expected_shape", shape)
        setattr(param, "_litelinear_expected_dtype", dtype)
        setattr(param, "_litelinear_loaded", False)
        setattr(param, "weight_loader", _make_factor_weight_loader(name))
        layer.register_parameter(name, param)

    def _has_loaded_factors(self, layer: torch.nn.Module) -> bool:
        return all(
            getattr(getattr(layer, name, None), "_litelinear_loaded", False)
            for name in self.FACTOR_NAMES
        )

    def _process_loaded_factors(self, layer: torch.nn.Module) -> None:
        LiteLinear = _load_litelinear_class()
        A = layer.A
        B = layer.B
        lite_layer = _make_lite_linear(
            LiteLinear,
            in_features=B.shape[1],
            out_features=A.shape[0],
            rank=B.shape[0],
            device=A.device,
            dtype=A.dtype,
        )
        state_dict = {name: getattr(layer, name).detach() for name in self.FACTOR_NAMES}
        lite_layer.load_state_dict(state_dict, strict=True)
        lite_layer.eval()

        if self.prefix:
            setattr(lite_layer, "_lite_key", self.prefix)

        layer.register_parameter("weight", None)
        self._clear_factor_parameters(layer)
        object.__setattr__(layer, "_lite_linear_module", lite_layer)

    def _clear_factor_parameters(self, layer: torch.nn.Module) -> None:
        for name in self.FACTOR_NAMES:
            layer.register_parameter(name, None)


def _load_litelinear_class():
    try:
        module = importlib.import_module("lite_linear")
    except ImportError as exc:
        raise ImportError(
            "`--quantization litelinear` requires the optional `lite_linear` "
            "package. Install a LiteLinear wheel/package in the active conda "
            "environment, or run without `--quantization litelinear`."
        ) from exc

    try:
        return module.LiteLinear
    except AttributeError as exc:
        raise ImportError(
            "The `lite_linear` package does not export LiteLinear."
        ) from exc


def _make_factor_weight_loader(factor_name: str):
    def factor_weight_loader(
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_id: Optional[int | str] = None,
    ) -> None:
        if shard_id is not None:
            raise ValueError(
                "LiteLinear factor checkpoints must store fused factor tensors "
                f"directly on the LiteLinear module. Got shard_id={shard_id!r} "
                f"while loading factor {factor_name}."
            )
        expected_shape = getattr(param, "_litelinear_expected_shape", None)
        expected_dtype = getattr(param, "_litelinear_expected_dtype", None)
        if expected_shape is not None and tuple(loaded_weight.shape) != expected_shape:
            raise ValueError(
                f"Attempted to load LiteLinear factor {factor_name} with shape "
                f"{loaded_weight.size()} into expected shape {expected_shape}."
            )
        if isinstance(param, UninitializedParameter):
            param.materialize(
                tuple(loaded_weight.shape),
                device=param.device,
                dtype=expected_dtype or loaded_weight.dtype,
            )
        if param.numel() == 1 and loaded_weight.numel() == 1:
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load LiteLinear factor {factor_name} with shape "
                f"{loaded_weight.size()} into parameter {param.size()}"
            )
            param.data.copy_(loaded_weight)
        setattr(param, "_litelinear_loaded", True)

    return factor_weight_loader


def maybe_resolve_litelinear_patched_weights(
    hf_weights_files: List[str],
    model_loader_extra_config: Dict[str, Any],
) -> List[str]:
    checkpoint_mode = _get_checkpoint_mode(model_loader_extra_config)
    if checkpoint_mode in ("", "dense", "none"):
        return hf_weights_files
    if checkpoint_mode not in ("online_patch", "strict"):
        raise ValueError(
            "litelinear_checkpoint_mode must be one of 'dense', "
            f"'online_patch', or 'strict', got {checkpoint_mode!r}."
        )

    try:
        from lite_linear.checkpoint_patch_core import (
            count_patchable_pairs_for_checkpoint,
            default_cache_root,
            resolve_or_build_patched_checkpoint,
            resolve_strict_checkpoint_path,
        )
        from lite_linear.patched_checkpoint import get_safe_open_for_patch_build
    except ImportError as exc:
        raise ImportError(
            "LiteLinear checkpoint patching requires a lite_linear package "
            "with checkpoint_patch_core support."
        ) from exc

    safe_open_fn = get_safe_open_for_patch_build()
    cache_root = Path(
        model_loader_extra_config.get("litelinear_patch_cache_root")
        or default_cache_root()
    )
    patch_config_path = model_loader_extra_config.get("litelinear_patch_config_path")
    patch_config_path = Path(patch_config_path) if patch_config_path else None
    rank = int(model_loader_extra_config.get("litelinear_rank", 64))
    tag = str(model_loader_extra_config.get("litelinear_patch_tag", "nocalib"))
    copy_original = bool(
        model_loader_extra_config.get("litelinear_patch_copy_original", True)
    )
    force_rebuild = bool(
        model_loader_extra_config.get("litelinear_patch_force_rebuild", False)
    )

    resolved_files: List[str] = []
    for file_name in hf_weights_files:
        src_path = Path(file_name)
        patchable_count = count_patchable_pairs_for_checkpoint(
            src_path,
            safe_open_fn=safe_open_fn,
            patch_config_path=patch_config_path,
        )
        if patchable_count == 0:
            resolved_files.append(file_name)
            continue

        if checkpoint_mode == "online_patch":
            resolved_path = resolve_or_build_patched_checkpoint(
                src_path,
                rank=rank,
                tag=tag,
                cache_root=cache_root,
                copy_original=copy_original,
                force_rebuild=force_rebuild,
                safe_open_fn=safe_open_fn,
                log_prefix="[SGLang LiteLinear]",
                patch_config_path=patch_config_path,
            )
        else:
            resolved_path = resolve_strict_checkpoint_path(
                src_path,
                rank=rank,
                tag=tag,
                cache_root=cache_root,
                safe_open_fn=safe_open_fn,
                log_prefix="[SGLang LiteLinear]",
                patch_config_path=patch_config_path,
            )
        resolved_files.append(str(resolved_path))

    return resolved_files


def _get_checkpoint_mode(model_loader_extra_config: Dict[str, Any]) -> str:
    checkpoint_mode = str(
        model_loader_extra_config.get("litelinear_checkpoint_mode", "dense")
    ).strip()
    if model_loader_extra_config.get("litelinear_online_patch", False):
        checkpoint_mode = "online_patch"
    if model_loader_extra_config.get("litelinear_strict_patch", False):
        checkpoint_mode = "strict"
    return checkpoint_mode


def _make_lite_linear(
    LiteLinear,
    in_features: int,
    out_features: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
):
    kwargs = {
        "bias": False,
        "rank": rank,
        "device": device,
        "dtype": dtype,
    }
    try:
        signature = inspect.signature(LiteLinear.__init__)
    except (TypeError, ValueError):
        lite_layer = LiteLinear(in_features, out_features, **kwargs)
    else:
        parameters = signature.parameters
        has_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        supported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if has_kwargs or key in parameters
        }
        lite_layer = LiteLinear(in_features, out_features, **supported_kwargs)

    return lite_layer.to(device=device, dtype=dtype)
