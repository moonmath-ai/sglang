"""Unit tests for the optional LiteLinear quantization hook."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeLiteLinear(nn.Module):
    instances = []

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        rank: int = 64,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.materialized = False
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)
        self.instances.append(self)

    def materialize_from_weight(self):
        self.materialized = True

    def load_state_dict(self, state_dict, strict=True):
        self.register_buffer("A", state_dict["A"].detach().clone())
        self.register_buffer("B", state_dict["B"].detach().clone())
        self.register_buffer("Q_fp8", state_dict["Q_fp8"].detach().clone())
        self.register_buffer(
            "Q_scale_inv", state_dict["Q_scale_inv"].detach().clone()
        )
        self.register_parameter("weight", None)
        self.materialized = True
        return nn.modules.module._IncompatibleKeys([], [])

    def forward(self, x):
        if self.weight is not None:
            return F.linear(x, self.weight, self.bias)
        weight = self.A.float() @ self.B.float()
        weight = weight + self.Q_fp8.float() * self.Q_scale_inv.float()
        return F.linear(x.float(), weight, self.bias).to(dtype=x.dtype)


def fake_litelinear_module():
    module = types.ModuleType("lite_linear")
    module.LiteLinear = FakeLiteLinear
    return module


class TestLiteLinearConfig(CustomTestCase):
    def test_quantization_registry_contains_litelinear(self):
        from sglang.srt.layers.quantization import get_quantization_config
        from sglang.srt.layers.quantization.litelinear import LiteLinearConfig

        self.assertIs(get_quantization_config("litelinear"), LiteLinearConfig)

    def test_model_loader_extra_config_selects_factor_loading(self):
        from sglang.srt.layers.quantization.litelinear import LiteLinearConfig

        config = LiteLinearConfig(rank=16)
        self.assertIn(
            "litelinear_checkpoint_mode",
            LiteLinearConfig.get_model_loader_extra_config_keys(),
        )

        config.update_from_model_loader_extra_config(
            {
                "litelinear_checkpoint_mode": "strict",
                "litelinear_rank": 32,
            }
        )

        self.assertFalse(config.load_dense_weight)
        self.assertEqual(config.rank, 32)

    def test_default_policy_uses_shape_not_name(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.litelinear import (
            LiteLinearConfig,
            LiteLinearMethod,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        config = LiteLinearConfig(min_input_size=1)
        mlp = ReplicatedLinear(
            4,
            12,
            quant_config=config,
            prefix="model.layers.0.mlp.custom_expansion",
        )
        attn = ReplicatedLinear(
            4,
            4,
            quant_config=config,
            prefix="model.layers.0.self_attn.q_proj",
        )

        self.assertIsInstance(mlp.quant_method, LiteLinearMethod)
        self.assertIsInstance(attn.quant_method, UnquantizedLinearMethod)

    def test_default_policy_supports_contraction_projection(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.litelinear import (
            LiteLinearConfig,
            LiteLinearMethod,
        )

        layer = ReplicatedLinear(
            12,
            4,
            quant_config=LiteLinearConfig(min_input_size=1),
            prefix="model.layers.0.mlp.down_proj",
        )

        self.assertIsInstance(layer.quant_method, LiteLinearMethod)

    def test_target_patterns_are_optional_extra_filter(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.litelinear import LiteLinearConfig
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        layer = ReplicatedLinear(
            4,
            12,
            quant_config=LiteLinearConfig(
                min_input_size=1,
                target_patterns=[r"(?:^|\.)gate_up_proj$"],
            ),
            prefix="model.layers.0.mlp.custom_expansion",
        )

        self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)

    def test_default_policy_skips_too_large_output_ratio(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.litelinear import LiteLinearConfig
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        layer = ReplicatedLinear(
            4,
            64,
            quant_config=LiteLinearConfig(min_input_size=1),
            prefix="model.layers.0.lm_head_like",
        )

        self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)

    def test_default_policy_skips_small_layers(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.litelinear import LiteLinearConfig
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        layer = ReplicatedLinear(
            2048,
            11264,
            quant_config=LiteLinearConfig(),
            prefix="model.layers.0.mlp.custom_expansion",
        )

        self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)

    def test_ignored_layer_uses_unquantized_method(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.litelinear import LiteLinearConfig
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        config = LiteLinearConfig(
            ignored_layers=["model.layers.0.mlp.gate_up_proj"],
            min_input_size=1,
        )
        layer = ReplicatedLinear(
            4,
            12,
            quant_config=config,
            prefix="model.layers.0.mlp.gate_up_proj",
        )

        self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)


class TestLiteLinearMethod(CustomTestCase):
    def setUp(self):
        FakeLiteLinear.instances.clear()

    def test_missing_optional_package_raises_clear_error(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.litelinear import LiteLinearConfig

        layer = ReplicatedLinear(
            4,
            12,
            quant_config=LiteLinearConfig(min_input_size=1),
            prefix="model.layers.0.mlp.gate_up_proj",
        )

        with patch.dict(sys.modules, {"lite_linear": None}):
            with self.assertRaisesRegex(ImportError, "requires the optional"):
                layer.quant_method.process_weights_after_loading(layer)

    def test_process_weights_and_apply_dispatch_through_litelinear(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.litelinear import LiteLinearConfig

        layer = ReplicatedLinear(
            4,
            8,
            bias=True,
            quant_config=LiteLinearConfig(
                rank=8,
                min_input_size=1,
                min_output_ratio=1,
                target_patterns=[r"(?:^|\.)down_proj$"],
            ),
            prefix="model.layers.0.mlp.down_proj",
        )
        with torch.no_grad():
            layer.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [2.0, 3.0, 4.0, 5.0],
                        [3.0, 4.0, 5.0, 6.0],
                        [4.0, 5.0, 6.0, 7.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [6.0, 7.0, 8.0, 9.0],
                        [7.0, 8.0, 9.0, 10.0],
                        [8.0, 9.0, 10.0, 11.0],
                    ]
                )
            )
            layer.bias.copy_(torch.arange(8, dtype=torch.float32) + 0.5)
            original_weight = layer.weight.detach().clone()

        module = fake_litelinear_module()
        with patch.dict(sys.modules, {"lite_linear": module}):
            layer.quant_method.process_weights_after_loading(layer)
            x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
            output = layer.quant_method.apply(layer, x, layer.bias)

        expected = F.linear(x, original_weight, layer.bias)
        torch.testing.assert_close(output, expected)
        self.assertIsNone(layer.weight)
        self.assertEqual(len(FakeLiteLinear.instances), 1)
        self.assertEqual(FakeLiteLinear.instances[0].rank, 8)
        self.assertTrue(FakeLiteLinear.instances[0].materialized)
        self.assertEqual(
            FakeLiteLinear.instances[0]._lite_key, "model.layers.0.mlp.down_proj"
        )

    def test_factor_checkpoint_path_loads_without_dense_weight(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.litelinear import LiteLinearConfig

        layer = ReplicatedLinear(
            4,
            8,
            quant_config=LiteLinearConfig(
                rank=2,
                min_input_size=1,
                load_dense_weight=False,
            ),
            prefix="model.layers.0.mlp.custom_expansion",
        )
        self.assertIsNone(layer.weight)

        A = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [0.0, 2.0],
                [2.0, 2.0],
                [3.0, 0.0],
                [0.0, 3.0],
            ]
        )
        B = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        )
        Q_fp8 = torch.zeros(8, 4, dtype=torch.float8_e4m3fn)
        Q_scale_inv = torch.tensor(1.0)

        layer.A.weight_loader(layer.A, A)
        layer.B.weight_loader(layer.B, B)
        layer.Q_fp8.weight_loader(layer.Q_fp8, Q_fp8)
        layer.Q_scale_inv.weight_loader(layer.Q_scale_inv, Q_scale_inv)

        module = fake_litelinear_module()
        with patch.dict(sys.modules, {"lite_linear": module}):
            layer.quant_method.process_weights_after_loading(layer)
            x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
            output = layer.quant_method.apply(layer, x)

        expected = F.linear(x, A @ B)
        torch.testing.assert_close(output, expected)
        self.assertIsNone(layer.weight)
        self.assertIsNone(layer.A)
        self.assertIsNone(layer.B)
        self.assertIsNone(layer.Q_fp8)
        self.assertIsNone(layer.Q_scale_inv)
        self.assertTrue(FakeLiteLinear.instances[0].materialized)

    def test_factor_checkpoint_path_requires_all_factors_without_dense_weight(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.litelinear import LiteLinearConfig

        layer = ReplicatedLinear(
            4,
            8,
            quant_config=LiteLinearConfig(
                rank=2,
                min_input_size=1,
                load_dense_weight=False,
            ),
            prefix="model.layers.0.mlp.custom_expansion",
        )

        with self.assertRaisesRegex(RuntimeError, "factor tensors were not loaded"):
            layer.quant_method.process_weights_after_loading(layer)

    def test_split_factor_checkpoint_raises_clear_error(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.litelinear import LiteLinearConfig

        layer = ReplicatedLinear(
            4,
            8,
            quant_config=LiteLinearConfig(rank=2, min_input_size=1),
            prefix="model.layers.0.mlp.custom_expansion",
        )

        with self.assertRaisesRegex(ValueError, "fused factor tensors"):
            layer.A.weight_loader(layer.A, torch.empty(8, 2), 0)

    def test_skip_bias_add_is_preserved(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.litelinear import LiteLinearConfig

        layer = ReplicatedLinear(
            2,
            4,
            bias=True,
            skip_bias_add=True,
            quant_config=LiteLinearConfig(
                min_input_size=1,
                min_output_ratio=1,
                target_patterns=[r"(?:^|\.)down_proj$"],
            ),
            prefix="model.layers.0.mlp.down_proj",
        )
        with torch.no_grad():
            layer.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [2.0, 0.0],
                        [0.0, 2.0],
                    ]
                )
            )
            layer.bias.copy_(torch.tensor([10.0, 20.0, 30.0, 40.0]))

        module = fake_litelinear_module()
        with patch.dict(sys.modules, {"lite_linear": module}):
            output, output_bias = layer(torch.tensor([[1.0, 2.0]]))

        torch.testing.assert_close(output, torch.tensor([[1.0, 2.0, 2.0, 4.0]]))
        torch.testing.assert_close(output_bias, layer.bias)


if __name__ == "__main__":
    unittest.main()
