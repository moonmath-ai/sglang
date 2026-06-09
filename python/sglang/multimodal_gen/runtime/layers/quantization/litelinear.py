from __future__ import annotations

import importlib
import inspect
import re
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
from sglang.multimodal_gen.runtime.models.parameter import ModelWeightParameter
from sglang.srt.layers.quantization.utils import is_layer_skipped


class LiteLinearConfig(QuantizationConfig):
    """Optional LiteLinear integration for diffusion transformer FFN layers."""

    CHECKPOINT_FORMAT_DENSE = "dense"
    CHECKPOINT_FORMAT_FACTORS = "factors"

    def __init__(
        self,
        rank: int = 64,
        target_patterns: list[str] | str | None = None,
        ignored_layers: list[str] | str | None = None,
        packed_modules_mapping: dict[str, list[str]] | None = None,
        checkpoint_format: str = CHECKPOINT_FORMAT_DENSE,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LiteLinear rank must be positive, got {rank}.")

        self.rank = rank
        self.checkpoint_format = self._normalize_checkpoint_format(checkpoint_format)
        self.target_patterns = self._normalize_string_list(
            target_patterns, default=[]
        )
        if not self.target_patterns:
            raise ValueError(
                "LiteLinear requires target_patterns regex entries. "
                "Layer selection is name-based, not shape-based."
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
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LiteLinearConfig":
        return cls(
            rank=int(cls.get_from_keys_or(config, ["rank", "litelinear_rank"], 64)),
            target_patterns=cls.get_from_keys_or(
                config, ["target_patterns", "litelinear_target_patterns"], None
            ),
            ignored_layers=cls.get_from_keys_or(
                config, ["ignored_layers", "modules_to_not_convert"], None
            ),
            packed_modules_mapping=cls.get_from_keys_or(
                config, ["packed_modules_mapping"], {}
            ),
            checkpoint_format=cls.get_from_keys_or(
                config,
                ["checkpoint_format", "litelinear_checkpoint_format"],
                cls.CHECKPOINT_FORMAT_DENSE,
            ),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if not isinstance(layer, LinearBase):
            return None

        if prefix in self.ignored_layers or is_layer_skipped(
            prefix,
            self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return UnquantizedLinearMethod()

        if not self._is_allowed_by_name(prefix):
            return UnquantizedLinearMethod()

        return LiteLinearMethod(self, prefix)

    def get_scaled_act_names(self) -> list[str]:
        return []

    def _is_allowed_by_name(self, prefix: str) -> bool:
        if not prefix:
            return False
        return any(pattern.search(prefix) for pattern in self._compiled_target_patterns)

    @staticmethod
    def _normalize_checkpoint_format(value: str) -> str:
        value = str(value).strip().lower()
        if value in ("", "dense", "none"):
            return LiteLinearConfig.CHECKPOINT_FORMAT_DENSE
        if value in ("factors", "factor"):
            return LiteLinearConfig.CHECKPOINT_FORMAT_FACTORS
        raise ValueError(
            "LiteLinear checkpoint_format must be either 'dense' or "
            f"'factors', got {value!r}."
        )

    @staticmethod
    def _normalize_string_list(
        value: str | list[str] | None,
        default: list[str],
    ) -> list[str]:
        if value is None:
            return default
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return list(value)


class LiteLinearMethod(LinearMethodBase):
    FACTOR_NAMES = ("A", "B", "Q_fp8", "Q_scale_inv")
    _LOADED_FACTORS_ATTR = "_litelinear_loaded_factors"
    _FACTOR_SHAPES_ATTR = "_litelinear_factor_shapes"

    def __init__(self, quant_config: LiteLinearConfig, prefix: str = "") -> None:
        self.quant_config = quant_config
        self.prefix = prefix

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)
        if (
            self.quant_config.checkpoint_format
            == LiteLinearConfig.CHECKPOINT_FORMAT_FACTORS
        ):
            layer.register_parameter("weight", None)
            self._register_factor_parameter(
                layer,
                "A",
                (output_size_per_partition, self.quant_config.rank),
                params_dtype,
            )
            self._register_factor_parameter(
                layer,
                "B",
                (self.quant_config.rank, input_size_per_partition),
                params_dtype,
            )
            self._register_factor_parameter(
                layer,
                "Q_fp8",
                (output_size_per_partition, input_size_per_partition),
                torch.float8_e4m3fn,
            )
            self._register_factor_parameter(layer, "Q_scale_inv", (), torch.float32)
            return

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_lite_linear_module", None) is not None:
            return

        if (
            self.quant_config.checkpoint_format
            == LiteLinearConfig.CHECKPOINT_FORMAT_FACTORS
        ):
            self._process_loaded_factors(layer)
            return

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

        # Avoid registering this wrapper while model.named_modules() is iterating.
        object.__setattr__(layer, "_lite_linear_module", lite_layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
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

    def _init_factor_loading_state(
        self, layer: torch.nn.Module, name: str, shape: tuple[int, ...]
    ) -> None:
        if not hasattr(layer, self._LOADED_FACTORS_ATTR):
            object.__setattr__(layer, self._LOADED_FACTORS_ATTR, set())
            object.__setattr__(layer, self._FACTOR_SHAPES_ATTR, {})
        getattr(layer, self._FACTOR_SHAPES_ATTR)[name] = shape

    def _register_factor_parameter(
        self,
        layer: torch.nn.Module,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        self._init_factor_loading_state(layer, name, shape)
        param = Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        setattr(param, "weight_loader", _make_factor_weight_loader(name, layer))
        layer.register_parameter(name, param)

    def _process_loaded_factors(self, layer: torch.nn.Module) -> None:
        loaded_factors = getattr(layer, self._LOADED_FACTORS_ATTR, set())
        missing = [name for name in self.FACTOR_NAMES if name not in loaded_factors]
        if missing:
            input_size = getattr(layer, "input_size", None)
            output_size = getattr(layer, "output_size", None)
            raise RuntimeError(
                f"LiteLinear layer {self.prefix or '<unnamed>'} expects an "
                "offline factor checkpoint but did not load "
                f"(input_size={input_size}, output_size={output_size}): "
                f"{', '.join(missing)}."
            )

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
        for name in self.FACTOR_NAMES:
            layer.register_parameter(name, None)
        object.__setattr__(layer, "_lite_linear_module", lite_layer)


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


def _make_factor_weight_loader(factor_name: str, layer: torch.nn.Module):
    def factor_weight_loader(
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_id: int | str | None = None,
    ) -> None:
        if shard_id is not None:
            raise ValueError(
                "LiteLinear offline factor checkpoints must store fused factor "
                f"tensors directly on the LiteLinear module. Got shard_id={shard_id!r} "
                f"while loading factor {factor_name}."
            )

        factor_shapes = getattr(layer, LiteLinearMethod._FACTOR_SHAPES_ATTR, {})
        expected_shape = factor_shapes.get(factor_name)
        if expected_shape is not None and tuple(loaded_weight.shape) != expected_shape:
            raise ValueError(
                f"Attempted to load LiteLinear factor {factor_name} with shape "
                f"{loaded_weight.size()} into expected shape {expected_shape}."
            )

        if param.numel() == 1 and loaded_weight.numel() == 1:
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load LiteLinear factor {factor_name} with shape "
                f"{loaded_weight.size()} into parameter {param.size()}"
            )
            param.data.copy_(loaded_weight)
        getattr(layer, LiteLinearMethod._LOADED_FACTORS_ATTR).add(factor_name)

    return factor_weight_loader


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
        supported_kwargs = {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
        lite_layer = LiteLinear(in_features, out_features, **supported_kwargs)
    except (TypeError, ValueError):
        lite_layer = LiteLinear(in_features, out_features, **kwargs)

    return lite_layer.to(device=device, dtype=dtype)
