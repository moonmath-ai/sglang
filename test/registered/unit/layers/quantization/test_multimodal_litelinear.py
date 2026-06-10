"""Unit tests for the multimodal LiteLinear quantization hook."""

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


class TestMultimodalLiteLinearConfig(CustomTestCase):
    def test_quantization_registry_contains_litelinear(self):
        from sglang.multimodal_gen.runtime.layers.quantization import (
            get_quantization_config,
        )
        from sglang.multimodal_gen.runtime.layers.quantization.litelinear import (
            LiteLinearConfig,
        )

        self.assertIs(get_quantization_config("litelinear"), LiteLinearConfig)

    def test_from_config_selects_factor_checkpoint_format(self):
        from sglang.multimodal_gen.runtime.layers.quantization.litelinear import (
            LiteLinearConfig,
        )

        config = LiteLinearConfig.from_config(
            {
                "quant_method": "litelinear",
                "checkpoint_format": "factors",
                "rank": 32,
                "target_patterns": [
                    r"transformer_blocks\..*(audio_)?ff\.proj_(in|out)$"
                ],
            }
        )

        self.assertEqual(config.checkpoint_format, "factors")
        self.assertEqual(config.rank, 32)

    def test_explicit_quantization_uses_litelinear_checkpoint_config(self):
        from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
            _resolve_quant_config,
        )

        config = _resolve_quant_config(
            hf_config={
                "quantization_config": {
                    "quant_method": "litelinear",
                    "checkpoint_format": "factors",
                    "rank": 32,
                    "target_patterns": [
                        r"transformer_blocks\..*(audio_)?ff\.proj_(in|out)$"
                    ],
                }
            },
            server_args=types.SimpleNamespace(
                quantization="litelinear",
                transformer_weights_path=None,
            ),
            safetensors_list=[],
            component_model_path="/tmp",
        )

        self.assertEqual(config.checkpoint_format, "factors")
        self.assertEqual(config.rank, 32)

    def test_transformer_weights_path_resolves_litelinear_config(self):
        import json
        import tempfile

        from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
            _resolve_quant_config,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = f"{tmpdir}/config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "quantization_config": {
                            "quant_method": "litelinear",
                            "checkpoint_format": "factors",
                            "rank": 32,
                            "target_patterns": [
                                r"transformer_blocks\..*(audio_)?ff\.proj_(in|out)$"
                            ],
                        }
                    },
                    f,
                )

            config = _resolve_quant_config(
                hf_config={},
                server_args=types.SimpleNamespace(
                    quantization=None,
                    transformer_weights_path=tmpdir,
                    pipeline_config=types.SimpleNamespace(
                        dit_config=types.SimpleNamespace(
                            arch_config=types.SimpleNamespace(
                                param_names_mapping={},
                                reverse_param_names_mapping=None,
                            )
                        )
                    ),
                ),
                safetensors_list=[],
                component_model_path="/tmp/component",
            )

        self.assertEqual(config.checkpoint_format, "factors")
        self.assertEqual(config.rank, 32)

    def test_process_weights_and_apply_dispatch_through_litelinear(self):
        from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
        from sglang.multimodal_gen.runtime.layers.quantization.litelinear import (
            LiteLinearConfig,
        )

        FakeLiteLinear.instances.clear()
        layer = ReplicatedLinear(
            4,
            8,
            bias=False,
            quant_config=LiteLinearConfig(
                rank=8,
                target_patterns=[r"transformer_blocks\..*\.ff\.proj_in$"],
            ),
            prefix="transformer_blocks.0.ff.proj_in",
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
            original_weight = layer.weight.detach().clone()

        module = fake_litelinear_module()
        with patch.dict(sys.modules, {"lite_linear": module}):
            layer.quant_method.process_weights_after_loading(layer)
            x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
            output, _ = layer(x)

        expected = F.linear(x, original_weight)
        torch.testing.assert_close(output, expected)
        self.assertIsNone(layer.weight)
        self.assertEqual(len(FakeLiteLinear.instances), 1)
        self.assertEqual(FakeLiteLinear.instances[0].rank, 8)
        self.assertTrue(FakeLiteLinear.instances[0].materialized)

    def test_factor_checkpoint_path_loads_without_dense_weight(self):
        from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
        from sglang.multimodal_gen.runtime.layers.quantization.litelinear import (
            LiteLinearConfig,
        )

        FakeLiteLinear.instances.clear()
        layer = ReplicatedLinear(
            4,
            8,
            bias=False,
            quant_config=LiteLinearConfig(
                rank=2,
                checkpoint_format="factors",
                target_patterns=[r"transformer_blocks\..*\.ff\.proj_in$"],
            ),
            prefix="transformer_blocks.0.ff.proj_in",
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
            output, _ = layer(x)

        expected = F.linear(x, A @ B)
        torch.testing.assert_close(output, expected)
        self.assertIsNone(layer.weight)
        self.assertIsNone(layer.A)
        self.assertIsNone(layer.B)
        self.assertIsNone(layer.Q_fp8)
        self.assertIsNone(layer.Q_scale_inv)
        self.assertTrue(FakeLiteLinear.instances[0].materialized)


    def test_requires_target_patterns(self):
        from sglang.multimodal_gen.runtime.layers.quantization.litelinear import (
            LiteLinearConfig,
        )

        with self.assertRaisesRegex(ValueError, "target_patterns"):
            LiteLinearConfig()

    def test_target_patterns_select_layers_by_prefix(self):
        from sglang.multimodal_gen.runtime.layers.linear import (
            ReplicatedLinear,
            UnquantizedLinearMethod,
        )
        from sglang.multimodal_gen.runtime.layers.quantization.litelinear import (
            LiteLinearConfig,
            LiteLinearMethod,
        )

        config = LiteLinearConfig(
            target_patterns=[r"transformer_blocks\..*(audio_)?ff\.proj_(in|out)$"]
        )
        ff = ReplicatedLinear(
            4,
            12,
            quant_config=config,
            prefix="transformer_blocks.0.ff.proj_in",
        )
        attn = ReplicatedLinear(
            4,
            4,
            quant_config=config,
            prefix="transformer_blocks.0.attn1.to_q",
        )

        self.assertIsInstance(ff.quant_method, LiteLinearMethod)
        self.assertIsInstance(attn.quant_method, UnquantizedLinearMethod)


if __name__ == "__main__":
    unittest.main()
