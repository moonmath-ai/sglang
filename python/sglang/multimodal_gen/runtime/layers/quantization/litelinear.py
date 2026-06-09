from __future__ import annotations

import importlib
import inspect
import re
from typing import Any

import torch

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

    DEFAULT_MIN_INPUT_SIZE = 4096
    DEFAULT_MIN_OUTPUT_RATIO = 2.0
    DEFAULT_MAX_OUTPUT_RATIO = 12.0

    def __init__(
        self,
        rank: int = 64,
        target_patterns: list[str] | str | None = None,
        ignored_layers: list[str] | str | None = None,
        packed_modules_mapping: dict[str, list[str]] | None = None,
        min_input_size: int = DEFAULT_MIN_INPUT_SIZE,
        min_output_ratio: float = DEFAULT_MIN_OUTPUT_RATIO,
        max_output_ratio: float = DEFAULT_MAX_OUTPUT_RATIO,
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

        if not self._is_allowed_by_name(prefix) or not self._is_supported_shape(layer):
            return UnquantizedLinearMethod()

        return LiteLinearMethod(self, prefix)

    def get_scaled_act_names(self) -> list[str]:
        return []

    def _is_allowed_by_name(self, prefix: str) -> bool:
        if not self._compiled_target_patterns:
            return True
        if not prefix:
            return False
        return any(pattern.search(prefix) for pattern in self._compiled_target_patterns)

    def _is_supported_shape(self, layer: torch.nn.Module) -> bool:
        input_size = int(getattr(layer, "input_size", 0))
        output_size = int(getattr(layer, "output_size", 0))
        if input_size <= 0 or output_size <= 0:
            return False
        if min(input_size, output_size) < self.min_input_size:
            return False

        feature_ratio = max(output_size / input_size, input_size / output_size)
        return self.min_output_ratio <= feature_ratio <= self.max_output_ratio

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
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
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
