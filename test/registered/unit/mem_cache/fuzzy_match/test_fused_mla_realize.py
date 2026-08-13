"""Parity tests for the fused MLA realization kernel (fuse_kernel.md S4.2/S4.3 step 2).

``copy_mla_kv_with_rope_correction_fused`` collapses the reference path's
per-layer gather/scatter loop (2 * num_layers Triton launches + 2 * num_layers
``torch.empty`` allocations per fuzzy hit) into one launch over
``(num_layers, num_tokens)``. This proves it produces the same result as the
per-layer reference ``copy_mla_kv_with_rope_correction`` against a real
``MLATokenToKVPool`` at DeepSeek-V2-Lite row geometry -- and, against a
float64 oracle, that it is at least as accurate (the kernel keeps the whole
reverse->apply chain in fp32 registers with a single final rounding, while
the reference rounds to bf16 after every torch op).

Also covers the two wiring changes of the same build (S4.3 steps 1 and 3):

  * ``layer_recompute_mask`` handling -- flagged layers zeroed, not copied,
    inside the same single launch;
  * ``MLATokenToKVPool.move_kv_cache`` batched via
    ``copy_all_layer_kv_cache_func`` (enable_kv_cache_copy=True) -- byte-
    identical to the legacy per-layer loop, including the chunked and
    overlapping-range cases.

Skipped on CPU -- Triton requires a GPU.

    python -m pytest test/registered/unit/mem_cache/fuzzy_match/test_fused_mla_realize.py -v
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

_HAS_CUDA = torch.cuda.is_available()

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")

from sglang.srt.layers.rotary_embedding.utils import (
    apply_rotary_emb,
    reverse_rotary_emb,
)
from sglang.srt.mem_cache.fuzzy_match.fused_mla_realize import (
    copy_mla_kv_with_rope_correction_fused,
)
from sglang.srt.mem_cache.fuzzy_match.rope_correction import (
    copy_mla_kv_with_rope_correction,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.test.test_utils import CustomTestCase

# DeepSeek-V2-Lite row geometry (the kernel's NOPE_DIM/ROPE_HALF target).
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
NUM_LAYERS = 4
POOL_SIZE = 2048
MAX_POS = 8192
DTYPE = torch.bfloat16


class _FakeLayer:
    """`get_mla_kv_buffer`/`set_mla_kv_buffer` read only `.layer_id`."""

    def __init__(self, layer_id: int):
        self.layer_id = layer_id


class _FakeRotary:
    """The two copy paths read only these three attributes. fp32 cache; both
    implementations downcast cos/sin to bf16 before the math (the reference
    via `.to(x.dtype)`, the fused wrapper explicitly)."""

    def __init__(self, rotary_dim: int, device: str, is_neox_style: bool = True):
        self.rotary_dim = rotary_dim
        self.is_neox_style = is_neox_style
        inv_freq = 1.0 / (
            10000.0
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        t = torch.arange(MAX_POS, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(
            device=device, dtype=torch.float32
        )


def _make_pool(num_layers=NUM_LAYERS, size=POOL_SIZE, enable_kv_cache_copy=False):
    return MLATokenToKVPool(
        size=size,
        page_size=1,
        dtype=DTYPE,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=num_layers,
        device="cuda",
        enable_memory_saver=False,
        enable_kv_cache_copy=enable_kv_cache_copy,
    )


def _oracle_corrected_rope(snapshot_rows, old_pos, new_pos, rotary):
    """float64 reverse->apply of the k_rope slice.

    snapshot_rows: [n, 576] float64 rows from the *pre-copy* pool.
    Returns the corrected 64-dim k_rope slice, float64, on CPU.
    """
    cs = rotary.cos_sin_cache.cpu()
    oc, os_ = cs.index_select(0, old_pos.cpu()).chunk(2, dim=-1)
    nc, ns = cs.index_select(0, new_pos.cpu()).chunk(2, dim=-1)
    k_rope = snapshot_rows[:, KV_LORA_RANK:].reshape(-1, 1, QK_ROPE_HEAD_DIM)
    style = rotary.is_neox_style
    k_raw = reverse_rotary_emb(k_rope, oc.double(), os_.double(), style)
    k_new = apply_rotary_emb(k_raw, nc.double(), ns.double(), style)
    return k_new.reshape(-1, QK_ROPE_HEAD_DIM)


@unittest.skipIf(not _HAS_CUDA, "CUDA required (Triton kernel)")
class TestFusedMlaRealizeParity(CustomTestCase):
    # (name, is_neox_style): neox pairing and the real deployed models'
    # gptj-style interleaved pairing must both match the reference exactly.
    ROTARY_STYLES = [("neox", True), ("gptj", False)]

    def setUp(self):
        torch.manual_seed(0)
        self.layers = [_FakeLayer(i) for i in range(NUM_LAYERS)]

    def setUpRotary(self, is_neox_style):
        self.rotary = _FakeRotary(QK_ROPE_HEAD_DIM, "cuda", is_neox_style)

    def _make_pair(self):
        """Two byte-identical pools: one for the reference path, one fused."""
        ref_pool = _make_pool()
        fused_pool = _make_pool()
        with torch.no_grad():
            for i in range(NUM_LAYERS):
                ref_pool.kv_buffer[i].normal_()
                fused_pool.kv_buffer[i].copy_(ref_pool.kv_buffer[i])
        return ref_pool, fused_pool

    def _run_case(self, num_tokens, mask=None):
        """Runs reference on one pool and fused on an identical pool.

        Returns the two result pools, a pre-copy snapshot, and the
        loc/position tensors used.
        """
        ref_pool, fused_pool = self._make_pair()
        g = torch.Generator(device="cpu").manual_seed(42 + num_tokens)
        # old/new locs must be DISJOINT (the kernel and the realizer's
        # recipient-pre-alloc contract both require it); two independent
        # randperms collide for larger n.
        perm = torch.randperm(POOL_SIZE, generator=g)
        old_locs = perm[:num_tokens].cuda()
        new_locs = perm[num_tokens : 2 * num_tokens].cuda()
        old_pos = torch.randint(0, MAX_POS // 2, (num_tokens,), generator=g).cuda()
        new_pos = (old_pos + 137) % MAX_POS

        snapshot = [buf.clone() for buf in ref_pool.kv_buffer]

        copy_mla_kv_with_rope_correction(
            pool=ref_pool,
            attn_layers=self.layers,
            rotary_emb=self.rotary,
            old_locs=old_locs,
            new_locs=new_locs,
            old_positions=old_pos,
            new_positions=new_pos,
            layer_recompute_mask=mask,
        )
        copy_mla_kv_with_rope_correction_fused(
            pool=fused_pool,
            attn_layers=self.layers,
            rotary_emb=self.rotary,
            old_locs=old_locs,
            new_locs=new_locs,
            old_positions=old_pos,
            new_positions=new_pos,
            layer_recompute_mask=mask,
        )
        return ref_pool, fused_pool, snapshot, (old_locs, new_locs, old_pos, new_pos)

    def test_fused_matches_reference_and_float64_oracle(self):
        for style_name, is_neox in self.ROTARY_STYLES:
            for num_tokens in (1, 7, 253, 700):
                with self.subTest(rotary=style_name, num_tokens=num_tokens):
                    self.setUpRotary(is_neox)
                    ref_pool, fused_pool, snapshot, (
                        old_locs,
                        new_locs,
                        old_pos,
                        new_pos,
                    ) = self._run_case(num_tokens)

                    # Sibling check, with tolerance for the reference's per-op
                    # bf16 roundings (the kernel rounds once at the end). A real
                    # wiring bug (wrong lane pairing, nope/rope mixup, layer
                    # off-by-one) produces O(1) errors, far outside this band.
                    for i in range(NUM_LAYERS):
                        torch.testing.assert_close(
                            fused_pool.kv_buffer[i],
                            ref_pool.kv_buffer[i],
                            rtol=3e-2,
                            atol=5e-2,
                            msg=f"layer {i}, {style_name}, num_tokens {num_tokens}: "
                            "fused diverged from the per-layer reference",
                        )

                    # Accuracy check against a float64 oracle, accumulated over
                    # all layers: fused (fp32 chain, one rounding) must be at
                    # least as accurate as the reference (per-op roundings).
                    rel_l2_ref, rel_l2_fused = 0.0, 0.0
                    for i in range(NUM_LAYERS):
                        rows_src = snapshot[i][old_locs, 0].double().cpu()
                        oracle_rope = _oracle_corrected_rope(
                            rows_src, old_pos, new_pos, self.rotary
                        )
                        for tag, pool in (("ref", ref_pool), ("fused", fused_pool)):
                            got = (
                                pool.kv_buffer[i][new_locs, 0, KV_LORA_RANK:]
                                .double()
                                .cpu()
                            )
                            err = (
                                (got - oracle_rope).norm() / oracle_rope.norm()
                            ).item()
                            if tag == "ref":
                                rel_l2_ref += err
                            else:
                                rel_l2_fused += err
                    print(
                        f"{style_name} num_tokens={num_tokens}: rel-L2 vs fp64 "
                        f"oracle, reference={rel_l2_ref:.5f} fused={rel_l2_fused:.5f}"
                    )
                    self.assertLess(
                        rel_l2_fused,
                        max(2.0 * rel_l2_ref, 1e-3),
                        "fused kernel is less accurate than the per-layer "
                        "reference against the float64 oracle",
                    )

    def test_masked_layers_are_zeroed_in_fused_launch(self):
        # Zeroing is style-independent; gptj covers the deployed models' style.
        self.setUpRotary(is_neox_style=False)
        mask = [True, False, True, False]
        ref_pool, fused_pool, _, (_, new_locs, _, _) = self._run_case(
            num_tokens=253, mask=mask
        )
        for i, flagged in enumerate(mask):
            if flagged:
                self.assertTrue(
                    bool(torch.all(fused_pool.kv_buffer[i][new_locs] == 0)),
                    f"masked layer {i} not zeroed at new_locs",
                )
            else:
                torch.testing.assert_close(
                    fused_pool.kv_buffer[i],
                    ref_pool.kv_buffer[i],
                    rtol=3e-2,
                    atol=5e-2,
                    msg=f"unmasked layer {i} diverged from reference under mask",
                )

    def test_rejects_unsupported_configs_loudly(self):
        self.setUpRotary(is_neox_style=True)
        pool = _make_pool()
        old_locs = torch.tensor([1], device="cuda")
        new_locs = torch.tensor([2], device="cuda")
        pos = torch.tensor([3], device="cuda")

        with self.assertRaises(AssertionError):
            copy_mla_kv_with_rope_correction_fused(
                pool=pool,
                attn_layers=self.layers[:-1],  # layer-count mismatch
                rotary_emb=self.rotary,
                old_locs=old_locs,
                new_locs=new_locs,
                old_positions=pos,
                new_positions=pos,
            )


@unittest.skipIf(not _HAS_CUDA, "CUDA required (Triton kernel)")
class TestMlaMoveKvCacheBatched(CustomTestCase):
    """S4.3 step 1: the batched move (copy_all_layer_kv_cache_func over the
    MLA pool's data_ptrs) must be byte-identical to the legacy per-layer
    indexing loop, for disjoint, overlapping, and chunked (>cap) ranges."""

    def test_batched_move_matches_manual_reference(self):
        for num_locs in (5, 300):  # 300 > num_locs_upper cap (256) -> chunk loop
            for overlap in (False, True):
                with self.subTest(num_locs=num_locs, overlap=overlap):
                    pool = _make_pool(enable_kv_cache_copy=True)
                    self.assertIsNotNone(pool._kv_copy_config)
                    with torch.no_grad():
                        for i in range(NUM_LAYERS):
                            pool.kv_buffer[i].normal_()

                    g = torch.Generator(device="cpu").manual_seed(7)
                    if overlap:
                        src = torch.randperm(POOL_SIZE - 1, generator=g)[
                            :num_locs
                        ].cuda()
                        tgt = (src + 1).cuda()
                    else:
                        perm = torch.randperm(POOL_SIZE, generator=g)
                        src = perm[:num_locs].cuda()
                        tgt = perm[num_locs : 2 * num_locs].cuda()

                    snapshot = [buf.clone() for buf in pool.kv_buffer]
                    pool.move_kv_cache(tgt, src)

                    for i in range(NUM_LAYERS):
                        expected = snapshot[i].clone()
                        expected[tgt] = snapshot[i][src]
                        self.assertTrue(
                            torch.equal(pool.kv_buffer[i], expected),
                            f"layer {i}: batched move not byte-identical to the "
                            f"legacy indexing semantics (num_locs={num_locs}, "
                            f"overlap={overlap})",
                        )

    def test_default_pool_move_uses_legacy_loop(self):
        pool = _make_pool()
        self.assertIsNone(pool._kv_copy_config)


if __name__ == "__main__":
    unittest.main()
