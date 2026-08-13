# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-forward realization of fuzzy-matched donor KV.

Donor K values in the pool carry RoPE for the donor's original positions.
Before the recipient's extend forward runs, :class:`FuzzyKVRealizer` copies
the donor KV into recipient-owned slots (pre-allocated by
``FuzzyRadixCache.match_prefix``), reversing the donor RoPE and applying it
at the recipient positions, then repoints ``req_to_token`` at the new slots.

Runs on the forward stream, invoked by ``ModelRunner`` (mirroring the
deferred mamba COW/clear pattern) so it stays outside CUDA-graph capture.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.fuzzy_match.rope_correction import (
    copy_kv_with_rope_correction,
    copy_mla_kv_with_rope_correction,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class FuzzyKVRealizer:
    """Copies donor KV into recipient slots with RoPE position correction."""

    def __init__(self, req_to_token_pool, token_to_kv_pool_allocator, model):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        pool = token_to_kv_pool_allocator.get_kvcache()
        if isinstance(pool, HybridLinearKVPool) and not pool.use_mla:
            # Hybrid full-attention + linear-attention models (e.g. Qwen3.5's
            # GatedDeltaNet layers) keep the full-attention layers' KV in a
            # nested pool (`full_kv_pool`), already indexed 0..N-1 scoped to
            # just those layers — exactly what copy_kv_with_rope_correction's
            # `range(pool.layer_num)` loop expects, with no extra layer-id
            # remapping needed. The linear-attention layers have no
            # per-token K to correct at all (irreversible recurrent state,
            # not addressable K — research.md's X-factor test) and live
            # entirely outside this nested pool, so there's nothing else to
            # unwrap for them.
            pool = pool.full_kv_pool
        self.pool = pool
        # MHA pools index separate k_buffer/v_buffer lists directly by
        # layer_id. MLA pools store K/V as one fused per-layer latent
        # buffer behind get_mla_kv_buffer/set_mla_kv_buffer, which take
        # the layer's RadixAttention instance (for .layer_id) rather than
        # a raw int -- copy_mla_kv_with_rope_correction encapsulates that
        # difference so _realize_contiguous/_realize_segments don't need
        # to know about it.
        self.is_mla_pool = isinstance(self.pool, MLATokenToKVPool)
        self.mla_attn_layers = None
        if isinstance(self.pool, MHATokenToKVPool):
            self.pool_supported = True
        elif self.is_mla_pool:
            self.mla_attn_layers = self._resolve_mla_attn_layers(model)
            self.pool_supported = self.mla_attn_layers is not None
            if not self.pool_supported:
                logger.warning(
                    "[FUZZY] MLA model exposes no attn_mqa on every "
                    "self_attn layer; fuzzy realization disabled"
                )
        else:
            self.pool_supported = False
            logger.warning(
                "[FUZZY] KV pool %s is not MHA or MLA; fuzzy realization disabled",
                type(self.pool).__name__,
            )

        self.device = None
        if self.pool_supported:
            self.device = (
                self.pool.kv_buffer[0].device
                if self.is_mla_pool
                else self.pool.k_buffer[0].device
            )

        self.rotary_emb = self._resolve_rotary_emb(model)
        if self.rotary_emb is None:
            logger.warning(
                "[FUZZY] model exposes no rotary_emb on any layer; fuzzy "
                "realization disabled"
            )

        self.naive_reuse_hooks = None
        if envs.SGLANG_TEST_FUZZY_NAIVE_KV_REUSE.get():
            identity = lambda x, *args, **kwargs: x
            self.naive_reuse_hooks = {
                "apply_rotary_emb": identity,
                "reverse_rotary_emb": identity,
            }

    @staticmethod
    def _resolve_rotary_emb(model):
        # Model class layouts differ per architecture: some hold rotary_emb
        # under a self_attn submodule (qwen2.py's
        # layers[i].self_attn.rotary_emb), others hold it directly on the
        # decoder layer with no self_attn indirection at all (qwen3_5.py's
        # Qwen3_5AttentionDecoderLayer.rotary_emb). Hybrid architectures also
        # mix layer types across model.model.layers (Qwen3.5's GatedDeltaNet
        # layers expose neither), so probing layer 0 specifically is unsafe.
        # Scan for the first layer that actually has one — every
        # full-attention layer in a given model shares the same
        # rope_theta/rotary_dim/max_position_embeddings, so any one match is
        # as good as any other.
        inner = getattr(model, "model", None)
        layers = getattr(inner, "layers", None)
        if not layers:
            return None
        for layer in layers:
            self_attn = getattr(layer, "self_attn", None)
            rotary_emb = (
                getattr(self_attn, "rotary_emb", None)
                if self_attn is not None
                else None
            )
            if rotary_emb is None:
                rotary_emb = getattr(layer, "rotary_emb", None)
            if rotary_emb is not None:
                return rotary_emb
        return None

    @staticmethod
    def _resolve_mla_attn_layers(model) -> Optional[List[RadixAttention]]:
        # get_mla_kv_buffer/set_mla_kv_buffer take the layer's RadixAttention
        # instance (only .layer_id is read off it), not a raw int.
        # DeepseekV2AttentionMLA.attn_mqa (not attn_mha, a separate
        # RadixAttention with an incompatible shape used for a different
        # code path) is the instance whose .layer_id flows into the real
        # forward pass's pool writes (forward_mla.py's absorb path calls
        # self.attn_mqa(...), which reaches
        # token_to_kv_pool.set_mla_kv_buffer(layer=self.attn_mqa, ...)
        # inside the attention backend). All-or-nothing: a layer missing
        # attn_mqa means this model isn't the DeepseekV2AttentionMLA shape
        # this was built for.
        inner = getattr(model, "model", None)
        layers = getattr(inner, "layers", None)
        if not layers:
            return None
        attn_layers = []
        for layer in layers:
            self_attn = getattr(layer, "self_attn", None)
            attn_mqa = (
                getattr(self_attn, "attn_mqa", None) if self_attn is not None else None
            )
            if attn_mqa is None:
                return None
            attn_layers.append(attn_mqa)
        return attn_layers

    def _copy_kv(
        self,
        *,
        old_locs: torch.Tensor,
        new_locs: torch.Tensor,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor,
        layer_recompute_mask: Optional[List[bool]],
    ) -> None:
        if self.is_mla_pool:
            if envs.SGLANG_FUZZY_REALIZE_FUSED.get():
                # Fused single-launch kernel (fuse_kernel.md S4.2). Imported
                # lazily so the default path (and CPU-only test environments,
                # per rope_correction.py's own lazy-import pattern) never
                # pulls Triton in.
                from sglang.srt.mem_cache.fuzzy_match.fused_mla_realize import (
                    copy_mla_kv_with_rope_correction_fused,
                )

                copy_mla_kv_with_rope_correction_fused(
                    pool=self.pool,
                    attn_layers=self.mla_attn_layers,
                    rotary_emb=self.rotary_emb,
                    old_locs=old_locs,
                    new_locs=new_locs,
                    old_positions=old_positions,
                    new_positions=new_positions,
                    layer_recompute_mask=layer_recompute_mask,
                )
                return
            copy_mla_kv_with_rope_correction(
                pool=self.pool,
                attn_layers=self.mla_attn_layers,
                rotary_emb=self.rotary_emb,
                old_locs=old_locs,
                new_locs=new_locs,
                old_positions=old_positions,
                new_positions=new_positions,
                layer_recompute_mask=layer_recompute_mask,
                **(self.naive_reuse_hooks or {}),
            )
        else:
            copy_kv_with_rope_correction(
                pool=self.pool,
                rotary_emb=self.rotary_emb,
                old_locs=old_locs,
                new_locs=new_locs,
                old_positions=old_positions,
                new_positions=new_positions,
                layer_recompute_mask=layer_recompute_mask,
                **(self.naive_reuse_hooks or {}),
            )

    def realize(self, fuzzy_reqs: List[Req]) -> None:
        """Realize every pending fuzzy match, then clear per-request state.

        Clearing (``fuzzy_realized_locs = None``, ``cache_fuzzy_matched_len
        = 0``) guarantees chunked-prefill re-entry, decode batches, and
        retract-and-resume never re-trigger a correction.
        """
        for req in fuzzy_reqs:
            try:
                self._realize_one(req)
            finally:
                req.fuzzy_realized_locs = None
                req.cache_fuzzy_matched_len = 0

    def _realize_one(self, req: Req) -> None:
        fuzzy_match_result = req.fuzzy_match_result
        num_fuzzy = req.cache_fuzzy_matched_len
        if fuzzy_match_result is None or num_fuzzy <= 0:
            return
        if not self.pool_supported or self.rotary_emb is None:
            self._free_realization_slots(req)
            return

        req_idx = req.req_pool_idx
        prefix_len = len(req.prefix_indices)

        if fuzzy_match_result.segments:
            self._realize_segments(
                req=req,
                req_idx=req_idx,
                segments=fuzzy_match_result.segments,
            )
        else:
            self._realize_contiguous(
                req=req,
                req_idx=req_idx,
                prefix_len=prefix_len,
                num_fuzzy=num_fuzzy,
                cached_start_pos=fuzzy_match_result.cached_start_pos,
                layer_recompute_mask=fuzzy_match_result.layer_recompute_mask,
            )

    def _realize_contiguous(
        self,
        req: Req,
        req_idx: int,
        prefix_len: int,
        num_fuzzy: int,
        cached_start_pos: int,
        layer_recompute_mask: Optional[List[bool]],
    ) -> None:
        exact_matched_len = prefix_len - num_fuzzy

        # No copy needed when donor positions already align with the target;
        # match_prefix allocated no slots in that case.
        if cached_start_pos == exact_matched_len:
            self._free_realization_slots(req)
            return

        new_fuzzy_locs = req.fuzzy_realized_locs
        if new_fuzzy_locs is None:
            # Pre-alloc contract violated; skipping is safer than allocating
            # here (would re-introduce a partial-commit window).
            logger.warning(
                "[FUZZY] contiguous correction: realized_locs missing; skipping"
            )
            return

        old_fuzzy_locs = self.req_to_token_pool.req_to_token[
            req_idx, exact_matched_len:prefix_len
        ]

        old_positions = torch.arange(
            cached_start_pos,
            cached_start_pos + num_fuzzy,
            device=self.device,
            dtype=torch.long,
        )
        new_positions = torch.arange(
            exact_matched_len,
            exact_matched_len + num_fuzzy,
            device=self.device,
            dtype=torch.long,
        )

        self._copy_kv(
            old_locs=old_fuzzy_locs,
            new_locs=new_fuzzy_locs,
            old_positions=old_positions,
            new_positions=new_positions,
            layer_recompute_mask=layer_recompute_mask,
        )

        # Point req_to_token at the new recipient-owned slots. The donor's
        # slots remain protected by its TreeNode lock_ref.
        self.req_to_token_pool.req_to_token[req_idx, exact_matched_len:prefix_len] = (
            new_fuzzy_locs
        )

        # The fuzzy span is now request-owned, not tree-owned; narrow the
        # protected prefix so duplicate-free ranges cover the realized slots.
        req.cache_protected_len = exact_matched_len

        logger.info(
            "[FUZZY] Realized %d fuzzy tokens (contiguous): copied donor KV "
            "with RoPE correction from positions [%d..%d] to [%d..%d]",
            num_fuzzy,
            cached_start_pos,
            cached_start_pos + num_fuzzy - 1,
            exact_matched_len,
            prefix_len - 1,
        )

    def _realize_segments(self, req: Req, req_idx: int, segments) -> None:
        """N:M alignment with scattered target positions.

        Each segment has its own donor locs and target positions; segments
        slice the pre-allocated ``fuzzy_realized_locs`` block in order.
        Before overwriting ``req_to_token`` at the target positions, the
        slots ``alloc_for_extend`` placed there are freed — otherwise they
        have no owner after the overwrite.
        """
        realized_locs = req.fuzzy_realized_locs
        if realized_locs is None:
            logger.warning(
                "[FUZZY] segments correction: realized_locs missing; skipping"
            )
            return

        req_to_token = self.req_to_token_pool.req_to_token
        device = self.device

        cursor = 0
        total_realized = 0
        for seg in segments:
            if seg.donor_kv_indices is None:
                # Providers must materialize NodeRef segments into
                # donor_kv_indices before the model executor runs.
                logger.warning("[FUZZY] segment without donor_kv_indices; skipping")
                continue

            target_positions = seg.target_positions.to(device).to(torch.long)
            donor_positions = seg.donor_positions.to(device).to(torch.long)
            donor_locs = seg.donor_kv_indices.to(device).to(torch.long)
            seg_len = seg.length if seg.length is not None else target_positions.numel()

            if cursor + seg_len > realized_locs.numel():
                logger.warning(
                    "[FUZZY] segment correction ran out of realized_locs; "
                    "cursor=%d seg_len=%d total=%d",
                    cursor,
                    seg_len,
                    realized_locs.numel(),
                )
                break

            new_locs = realized_locs[cursor : cursor + seg_len].to(torch.long)
            cursor += seg_len

            # Free the extend slots the scheduler placed at these target
            # positions before overwriting req_to_token.
            displaced = req_to_token[req_idx, target_positions].to(torch.int64)
            if displaced.numel() > 0:
                self.token_to_kv_pool_allocator.free(displaced)

            self._copy_kv(
                old_locs=donor_locs,
                new_locs=new_locs,
                old_positions=donor_positions,
                new_positions=target_positions,
                layer_recompute_mask=seg.layer_recompute_mask,
            )

            req_to_token[req_idx, target_positions] = new_locs.to(req_to_token.dtype)
            total_realized += seg_len

        logger.info(
            "[FUZZY] Realized %d fuzzy tokens (%d segments)",
            total_realized,
            len(segments),
        )

        if cursor < realized_locs.numel():
            self.token_to_kv_pool_allocator.free(realized_locs[cursor:])

    def _free_realization_slots(self, req: Req) -> None:
        if req.fuzzy_realized_locs is not None:
            self.token_to_kv_pool_allocator.free(req.fuzzy_realized_locs)
            req.fuzzy_realized_locs = None
