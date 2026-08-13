"""Unit tests for the MLA k_rope delta-rotation math under YaRN scaling —
no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.utils import (
    apply_rotary_emb,
    reverse_rotary_emb,
)
from sglang.srt.mem_cache.fuzzy_match.rope_correction import (
    _donor_target_cos_sin,
    copy_mla_kv_with_rope_correction,
)
from sglang.srt.runtime_context import get_context, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

HEAD_SIZE = 64  # DeepSeek-V2-Lite's qk_rope_head_dim
MAX_POSITION_EMBEDDINGS = 4096  # original_max_position_embeddings
ROPE_THETA = 10000
SCALING_FACTOR = 40  # YaRN "factor"
BETA_FAST = 32
BETA_SLOW = 1


def _make_rotary_emb(mscale, mscale_all_dim):
    return DeepseekScalingRotaryEmbedding(
        head_size=HEAD_SIZE,
        rotary_dim=HEAD_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        base=ROPE_THETA,
        is_neox_style=True,
        scaling_factor=SCALING_FACTOR,
        dtype=torch.float32,
        mscale=mscale,
        mscale_all_dim=mscale_all_dim,
        beta_fast=BETA_FAST,
        beta_slow=BETA_SLOW,
        device="cpu",
    )


def _cos_sin_at(rotary_emb, positions):
    cos_sin = rotary_emb.cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    return cos, sin


class _FakeAttnLayer:
    """Stands in for a real ``RadixAttention`` -- ``get_mla_kv_buffer``/
    ``set_mla_kv_buffer`` only ever read ``.layer_id`` off it."""

    def __init__(self, layer_id):
        self.layer_id = layer_id


class _FakeMlaPool:
    """CPU-only stand-in for ``MLATokenToKVPool``, matching its real
    combined-buffer layout (nope+rope packed into one per-layer tensor) so
    the masked-layer zero-write path (``pool.kv_buffer[...] = 0``) and the
    get/set accessors operate on the same underlying storage, same as the
    real pool."""

    def __init__(self, num_layers, num_slots, kv_lora_rank, qk_rope_head_dim):
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.start_layer = 0
        self.kv_buffer = [
            torch.randn(num_slots, 1, kv_lora_rank + qk_rope_head_dim)
            for _ in range(num_layers)
        ]

    def get_mla_kv_buffer(self, layer, loc):
        buf = self.kv_buffer[layer.layer_id - self.start_layer][loc]
        return buf[..., : self.kv_lora_rank], buf[..., self.kv_lora_rank :]

    def set_mla_kv_buffer(self, layer, loc, k_nope, k_rope):
        self.kv_buffer[layer.layer_id - self.start_layer][loc] = torch.cat(
            (k_nope, k_rope), dim=-1
        )


class TestMlaRopeCorrectionMscale(CustomTestCase):
    """The delta-rotation correction reverses RoPE at the donor position
    then reapplies it at the target position, exploiting rotation
    composability (R(a)*R(b)^-1 = R(a-b)). That composability assumes
    ``reverse_rotary_emb`` computes the true inverse of the forward
    rotation -- but it actually computes the forward matrix's *transpose*,
    which only equals the inverse when the rotation is orthonormal (true
    for plain RoPE, but not necessarily for YaRN's ``mscale``-scaled
    variant). This pins down the actual, quantified behavior in both the
    real deployed config (mscale == 1, transpose == inverse, no bias) and
    a mismatched one (mscale != 1, a uniform mscale**2 bias), rather than
    assuming "should be fine" from the deployed model happening to dodge
    the issue.
    """

    def setUp(self):
        self._saved_server_args = get_context()._server_args
        get_context().set_server_args(ServerArgs(model_path="dummy"))

    def tearDown(self):
        if self._saved_server_args is None:
            reset_context()
        else:
            get_context().set_server_args(self._saved_server_args)

    def test_deepseek_v2_lite_config_has_unit_mscale(self):
        """DeepSeek-V2-Lite's real config.json (mscale=mscale_all_dim=0.707)
        makes the YaRN amplitude ratio exactly 1.0 -- confirmed by
        computation, not assumed -- since mscale and mscale_all_dim feed
        into two calls of the same function with identical arguments. A
        future MLA checkpoint with mscale != mscale_all_dim would flip
        this, which is exactly the case
        test_mismatched_mscale_produces_scale_squared_bias below pins down.
        """
        rotary_emb = _make_rotary_emb(mscale=0.707, mscale_all_dim=0.707)
        self.assertAlmostEqual(rotary_emb.mscale, 1.0, places=6)

    def test_unit_mscale_round_trip_matches_fresh_rotation(self):
        """With mscale==1 (DeepSeek-V2-Lite's actual config), delta-rotation
        (reverse at the donor position, reapply at the target position)
        must exactly reproduce a direct one-shot rotation computed at the
        target position -- at more than one shift magnitude, since a bug
        that only manifests at large shifts (e.g. an index or precision
        issue) wouldn't be caught by a single offset.
        """
        rotary_emb = _make_rotary_emb(mscale=0.707, mscale_all_dim=0.707)
        torch.manual_seed(0)
        k_base = torch.randn(4, 1, HEAD_SIZE)

        for old_pos, new_pos in [(2, 6), (100, 5000)]:
            old_positions = torch.tensor([old_pos])
            new_positions = torch.tensor([new_pos])

            old_cos, old_sin = _cos_sin_at(rotary_emb, old_positions)
            new_cos, new_sin = _cos_sin_at(rotary_emb, new_positions)

            k_cached = apply_rotary_emb(k_base, old_cos, old_sin, True)
            k_recovered = reverse_rotary_emb(k_cached, old_cos, old_sin, True)
            k_corrected = apply_rotary_emb(k_recovered, new_cos, new_sin, True)

            k_expected = apply_rotary_emb(k_base, new_cos, new_sin, True)

            torch.testing.assert_close(k_corrected, k_expected, atol=1e-5, rtol=1e-5)

    def test_mismatched_mscale_produces_scale_squared_bias(self):
        """If a future MLA checkpoint's YaRN config has mscale !=
        mscale_all_dim (unlike DeepSeek-V2-Lite's), ``reverse_rotary_emb``'s
        "reverse" step computes the forward rotation's *transpose*, not its
        inverse -- these only coincide when the rotation is orthonormal
        (mscale == 1). This pins the actual, quantified consequence: a
        uniform multiplicative bias of ``mscale**2`` on every corrected
        vector, independent of the position shift. If the bias weren't
        shift-independent, no single fixed correction could compensate for
        it across all reuse distances -- this confirms it is.
        """
        rotary_emb = _make_rotary_emb(mscale=1.0, mscale_all_dim=0.5)
        self.assertNotAlmostEqual(rotary_emb.mscale, 1.0, places=3)
        expected_bias = rotary_emb.mscale**2

        torch.manual_seed(1)
        k_base = torch.randn(4, 1, HEAD_SIZE)

        for old_pos, new_pos in [(2, 6), (100, 5000)]:
            old_positions = torch.tensor([old_pos])
            new_positions = torch.tensor([new_pos])

            old_cos, old_sin = _cos_sin_at(rotary_emb, old_positions)
            new_cos, new_sin = _cos_sin_at(rotary_emb, new_positions)

            k_cached = apply_rotary_emb(k_base, old_cos, old_sin, True)
            k_recovered = reverse_rotary_emb(k_cached, old_cos, old_sin, True)
            k_corrected = apply_rotary_emb(k_recovered, new_cos, new_sin, True)

            k_expected = apply_rotary_emb(k_base, new_cos, new_sin, True)

            torch.testing.assert_close(
                k_corrected, k_expected * expected_bias, atol=1e-5, rtol=1e-5
            )

    def test_bf16_round_trip_rel_l2_error(self):
        """The paper reports rel-L2 error ~4.7e-3 for the round-trip
        correction in bf16 -- a rounding-precision number, not an
        algebraic-correctness one (float32 is exact per
        test_unit_mscale_round_trip_matches_fresh_rotation above). Reruns
        the same round trip in torch.bfloat16 using DeepSeek-V2-Lite's real
        deployed mscale (== 1.0) and reports/bounds
        ||corrected - expected||_2 / ||expected||_2 -- no other test in
        this file exercises bf16 dtype, so a regression here (e.g. an
        accidental extra downcast, a dropped upcast before accumulation)
        would otherwise go uncaught.
        """
        rotary_emb = _make_rotary_emb(mscale=0.707, mscale_all_dim=0.707)
        torch.manual_seed(2)
        k_base = torch.randn(4, 1, HEAD_SIZE, dtype=torch.bfloat16)

        for old_pos, new_pos in [(2, 6), (100, 5000)]:
            old_positions = torch.tensor([old_pos])
            new_positions = torch.tensor([new_pos])

            old_cos, old_sin = _cos_sin_at(rotary_emb, old_positions)
            new_cos, new_sin = _cos_sin_at(rotary_emb, new_positions)
            old_cos, old_sin = old_cos.to(torch.bfloat16), old_sin.to(torch.bfloat16)
            new_cos, new_sin = new_cos.to(torch.bfloat16), new_sin.to(torch.bfloat16)

            k_cached = apply_rotary_emb(k_base, old_cos, old_sin, True)
            k_recovered = reverse_rotary_emb(k_cached, old_cos, old_sin, True)
            k_corrected = apply_rotary_emb(k_recovered, new_cos, new_sin, True)
            k_expected = apply_rotary_emb(k_base, new_cos, new_sin, True)

            rel_l2 = (
                (k_corrected.float() - k_expected.float()).norm()
                / k_expected.float().norm()
            ).item()
            print(f"bf16 rel_l2 error (shift {old_pos}->{new_pos}): {rel_l2:.6f}")
            self.assertLess(
                rel_l2,
                2e-2,
                f"bf16 round-trip rel-L2 error {rel_l2} exceeds 2e-2 (10x the "
                f"paper's reported 4.7e-3) at shift {old_pos}->{new_pos}",
            )

    def test_naive_reuse_produces_large_rel_l2_error(self):
        """Naive reuse (paper's Table 2 "naive" column) copies donor K
        verbatim with no positional correction at all -- the same identity
        ``apply_rotary_emb``/``reverse_rotary_emb`` hooks
        ``FuzzyKVRealizer`` wires in under
        ``SGLANG_TEST_FUZZY_NAIVE_KV_REUSE`` (see ``realizer.py``). An
        end-to-end KL-divergence comparison of this against the corrected
        path is unreliable as a CI gate: probed empirically across shift
        magnitudes on both MLA targets, the gap between naive and
        corrected KL is small and non-monotonic for synthetic filler
        content (sometimes under 1.1x), consistent with the paper's own
        finding that naive reuse is statistically indistinguishable from
        PIC on at least one DeepSeek-V2-Lite workload (GovReport, ±0.02
        KL) -- MLA's compressed latent (``k_nope``, position-free) carries
        most of the attention-score magnitude, so corrupting only the
        ``k_rope`` slice doesn't always surface strongly in the final
        output distribution. What *is* algebraically guaranteed and
        deterministic is the K-vector-level error: identity hooks leave K
        rotated for the *donor's* position while it's read back as if it
        belonged to the *target* position, producing a large, shift-
        dependent rel-L2 error against the true target-position K -- this
        pins that down directly, without depending on how much a
        transformer's later layers happen to dampen it for any given
        input.
        """
        rotary_emb = _make_rotary_emb(mscale=0.707, mscale_all_dim=0.707)
        identity = lambda x, *args, **kwargs: x
        torch.manual_seed(3)
        k_base = torch.randn(4, 1, HEAD_SIZE)

        for old_pos, new_pos in [(32, 200), (100, 5000)]:
            old_positions = torch.tensor([old_pos])
            new_positions = torch.tensor([new_pos])

            old_cos, old_sin = _cos_sin_at(rotary_emb, old_positions)
            new_cos, new_sin = _cos_sin_at(rotary_emb, new_positions)

            k_cached = apply_rotary_emb(k_base, old_cos, old_sin, True)
            k_expected = apply_rotary_emb(k_base, new_cos, new_sin, True)

            k_naive = identity(
                identity(k_cached, old_cos, old_sin, True), new_cos, new_sin, True
            )
            k_corrected = apply_rotary_emb(
                reverse_rotary_emb(k_cached, old_cos, old_sin, True),
                new_cos,
                new_sin,
                True,
            )

            naive_rel_l2 = ((k_naive - k_expected).norm() / k_expected.norm()).item()
            corrected_rel_l2 = (
                (k_corrected - k_expected).norm() / k_expected.norm()
            ).item()
            print(
                f"shift {old_pos}->{new_pos}: naive_rel_l2={naive_rel_l2:.4f} "
                f"corrected_rel_l2={corrected_rel_l2:.2e}"
            )
            self.assertGreater(
                naive_rel_l2,
                0.5,
                f"naive-reuse rel-L2 error {naive_rel_l2} is surprisingly small "
                f"at shift {old_pos}->{new_pos} -- correction may not be as "
                "load-bearing as expected at this shift magnitude",
            )
            self.assertLess(
                corrected_rel_l2,
                1e-4,
                f"corrected rel-L2 error {corrected_rel_l2} is unexpectedly "
                f"large at shift {old_pos}->{new_pos}",
            )


class TestCopyMlaKvWithRopeCorrectionBatching(CustomTestCase):
    """``copy_mla_kv_with_rope_correction`` batches the rotation math across
    every active layer in one call instead of once per layer (profiling
    found the per-layer Python loop was ~95% CPU-side dispatch overhead --
    see the function's own docstring). This pins down that batching produces
    the exact same result as the straightforward per-layer reference
    computation -- the correctness claim a future change to the
    stack/scatter indexing (e.g. an off-by-one in which layer's slice in the
    stacked tensor goes to which pool buffer) could silently break without
    any other test catching it, since no existing test calls this function
    directly (only indirectly, through real E2E requests)."""

    def setUp(self):
        self._saved_server_args = get_context()._server_args
        get_context().set_server_args(ServerArgs(model_path="dummy"))

    def tearDown(self):
        if self._saved_server_args is None:
            reset_context()
        else:
            get_context().set_server_args(self._saved_server_args)

    def test_batched_rotation_matches_per_layer_reference(self):
        rotary_emb = _make_rotary_emb(mscale=0.707, mscale_all_dim=0.707)
        num_layers = 4
        num_slots = 20
        kv_lora_rank = 8

        torch.manual_seed(7)
        pool = _FakeMlaPool(num_layers, num_slots, kv_lora_rank, HEAD_SIZE)
        attn_layers = [_FakeAttnLayer(layer_id=i) for i in range(num_layers)]

        old_locs = torch.tensor([2, 5])
        new_locs = torch.tensor([10, 11])
        old_positions = torch.tensor([2, 100])
        new_positions = torch.tensor([6, 5000])

        # Reference: the same math applied one layer at a time, on an
        # identical starting pool.
        ref_pool = _FakeMlaPool(num_layers, num_slots, kv_lora_rank, HEAD_SIZE)
        for i in range(num_layers):
            ref_pool.kv_buffer[i].copy_(pool.kv_buffer[i])
        old_cos, old_sin, new_cos, new_sin = _donor_target_cos_sin(
            rotary_emb.cos_sin_cache, old_positions, new_positions
        )
        for layer in attn_layers:
            k_nope, k_rope = ref_pool.get_mla_kv_buffer(layer, old_locs)
            k_raw = reverse_rotary_emb(k_rope, old_cos, old_sin, True)
            k_rope_new = apply_rotary_emb(k_raw, new_cos, new_sin, True)
            ref_pool.set_mla_kv_buffer(layer, new_locs, k_nope, k_rope_new)

        copy_mla_kv_with_rope_correction(
            pool=pool,
            attn_layers=attn_layers,
            rotary_emb=rotary_emb,
            old_locs=old_locs,
            new_locs=new_locs,
            old_positions=old_positions,
            new_positions=new_positions,
        )

        for i in range(num_layers):
            torch.testing.assert_close(
                pool.kv_buffer[i],
                ref_pool.kv_buffer[i],
                msg=f"layer {i} diverged between batched and per-layer reference",
            )

    def test_masked_layers_are_zeroed_and_excluded_from_the_batch(self):
        """A ``layer_recompute_mask``-flagged layer must still be zeroed
        directly, not swept into the batched rotation -- pins down that the
        active-layer filtering (built for this refactor) doesn't
        accidentally include or corrupt masked layers."""
        rotary_emb = _make_rotary_emb(mscale=0.707, mscale_all_dim=0.707)
        num_layers = 3
        num_slots = 20
        kv_lora_rank = 8

        torch.manual_seed(8)
        pool = _FakeMlaPool(num_layers, num_slots, kv_lora_rank, HEAD_SIZE)
        attn_layers = [_FakeAttnLayer(layer_id=i) for i in range(num_layers)]

        old_locs = torch.tensor([2, 5])
        new_locs = torch.tensor([10, 11])
        old_positions = torch.tensor([2, 100])
        new_positions = torch.tensor([6, 5000])

        copy_mla_kv_with_rope_correction(
            pool=pool,
            attn_layers=attn_layers,
            rotary_emb=rotary_emb,
            old_locs=old_locs,
            new_locs=new_locs,
            old_positions=old_positions,
            new_positions=new_positions,
            layer_recompute_mask=[False, True, False],
        )

        self.assertTrue(torch.all(pool.kv_buffer[1][new_locs] == 0))
        self.assertFalse(torch.all(pool.kv_buffer[0][new_locs] == 0))
        self.assertFalse(torch.all(pool.kv_buffer[2][new_locs] == 0))


if __name__ == "__main__":
    unittest.main()
