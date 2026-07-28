"""Unit tests for the moonmath MLA attention backend eligibility logic.

These tests validate the _decode_eligible() gate without launching a server
or loading model weights. They check that the backend correctly routes to the
A16W8 kernel only for the supported case (fp8 KV, H<=16, MLA architecture,
pure decode) and falls back to aiter otherwise.

Requires: moonmath_attention installed (the backend imports it at __init__).
Requires: AMD ROCm (torch.float8_e4m3fnuz is a ROCm-only dtype).
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


def _make_layer(
    q_head_num=16,
    qk_head_dim=576,
    v_head_dim=512,
    tp_k_head_num=1,
    logit_cap=0,
    scaling=0.0884,
    k_scale=None,
):
    """Create a mock RadixAttention layer with MLA defaults."""
    layer = MagicMock()
    layer.tp_q_head_num = q_head_num
    layer.qk_head_dim = qk_head_dim
    layer.v_head_dim = v_head_dim
    layer.tp_k_head_num = tp_k_head_num
    layer.logit_cap = logit_cap
    layer.scaling = scaling
    layer.k_scale = k_scale
    return layer


def _make_fb(batch_size=4, forward_mode="decode", spec_info=None):
    """Create a mock ForwardBatch."""
    fb = MagicMock()
    fb.batch_size = batch_size
    fb.spec_info = spec_info
    mode = MagicMock()
    mode.is_decode.return_value = forward_mode == "decode"
    mode.is_target_verify.return_value = forward_mode == "target_verify"
    mode.is_extend.return_value = forward_mode == "extend"
    fb.forward_mode = mode
    return fb


def _make_backend(kv_cache_dtype=None, use_mla=True):
    """Create a MoonmathMLABackend with mocked dependencies."""
    with patch(
        "sglang.srt.layers.attention.aiter_backend.AiterAttnBackend.__init__"
    ), patch("moonmath_attention.mla") as mock_mla:
        mock_mla.mla_decode_a16w8_plan_parts_capped.return_value = 1
        from sglang.srt.layers.attention.moonmath_mla_backend import (
            MoonmathMLABackend,
        )

        runner = MagicMock()
        runner.use_mla = use_mla
        runner.device = torch.device("cuda")
        runner.model_config.context_len = 131072

        backend = MoonmathMLABackend.__new__(MoonmathMLABackend)
        backend._mla = mock_mla
        backend._mla_ok = use_mla
        backend._disabled = False
        backend._fp8_dtype = torch.float8_e4m3fnuz
        backend._fp8_kv = use_mla and (kv_cache_dtype == torch.float8_e4m3fnuz)
        backend._dec_parts = {}
        backend._mla_max_ctx = 131072
        backend._mla_seqlen_i32 = torch.zeros(8192, dtype=torch.int32, device="cpu")
        backend.kv_cache_dtype = kv_cache_dtype
        backend.forward_metadata = MagicMock()
        backend.forward_metadata.kv_indices = torch.zeros(1, dtype=torch.int32)
        backend.forward_metadata.kv_indptr = torch.zeros(1, dtype=torch.int32)
        return backend


class TestMoonmathMLAEligibility(unittest.TestCase):
    """Test _decode_eligible gates correctly for all input combinations."""

    def test_eligible_h16_fp8_decode(self):
        """The supported case: H=16, fp8 KV, pure decode, MLA dims."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        layer = _make_layer(q_head_num=16)
        fb = _make_fb(batch_size=4, forward_mode="decode")
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertTrue(backend._decode_eligible(q, layer, fb))

    def test_reject_h128(self):
        """H=128 (DSV3) should fall back to aiter."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        layer = _make_layer(q_head_num=128)
        fb = _make_fb(batch_size=4, forward_mode="decode")
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertFalse(backend._decode_eligible(q, layer, fb))

    def test_reject_bf16_kv(self):
        """bf16 KV should fall back to aiter (A16W8 requires fp8 KV)."""
        backend = _make_backend(kv_cache_dtype=torch.bfloat16)
        layer = _make_layer(q_head_num=16)
        fb = _make_fb(batch_size=4, forward_mode="decode")
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertFalse(backend._decode_eligible(q, layer, fb))

    def test_reject_extend(self):
        """Prefill/extend should fall back to aiter."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        layer = _make_layer(q_head_num=16)
        fb = _make_fb(batch_size=4, forward_mode="extend")
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertFalse(backend._decode_eligible(q, layer, fb))

    def test_reject_spec_verify(self):
        """Spec-verify should fall back to aiter."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        layer = _make_layer(q_head_num=16)
        fb = _make_fb(batch_size=4, spec_info=MagicMock())
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertFalse(backend._decode_eligible(q, layer, fb))

    def test_reject_disabled(self):
        """SGLANG_MOONMATH_MLA_DISABLE=1 should fall back to aiter."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        backend._disabled = True
        layer = _make_layer(q_head_num=16)
        fb = _make_fb(batch_size=4, forward_mode="decode")
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertFalse(backend._decode_eligible(q, layer, fb))

    def test_reject_wrong_dims(self):
        """Non-MLA dims (e.g. head_dim=128) should fall back to aiter."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz)
        layer = _make_layer(q_head_num=16, qk_head_dim=128, v_head_dim=128)
        fb = _make_fb(batch_size=4, forward_mode="decode")
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertFalse(backend._decode_eligible(q, layer, fb))

    def test_reject_non_mla(self):
        """Non-MLA model should fall back to aiter."""
        backend = _make_backend(kv_cache_dtype=torch.float8_e4m3fnuz, use_mla=False)
        layer = _make_layer(q_head_num=16)
        fb = _make_fb(batch_size=4, forward_mode="decode")
        q = torch.zeros(1, dtype=torch.bfloat16)
        self.assertFalse(backend._decode_eligible(q, layer, fb))


if __name__ == "__main__":
    unittest.main()
