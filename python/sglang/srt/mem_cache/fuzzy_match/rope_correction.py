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

"""Pure-Python RoPE correction helpers for fuzzy KV reuse.

The model executor's fuzzy paths (contiguous prefix-anchored, scattered
segments, and non-prefix-anchored match_block) all need to copy donor
K/V into recipient-owned slots with RoPE adjusted from the donor's
absolute positions to the recipient's. Keeping the per-layer copy in a
single helper rather than three inline loops avoids drift between paths
and gives unit tests a target they can exercise without importing the
full model runtime (which transitively pulls in torch extensions that
aren't present in pure-CPU CI).

Helpers:

* ``copy_kv_with_rope_correction`` - per-layer K/V copy that reverses
  RoPE at the donor position and reapplies it at the target position.
  When a ``layer_recompute_mask`` is provided, flagged layers are zeroed
  instead of copied (the bathtub-curve recompute path will produce fresh
  K/V for those layers in the next prefill pass).

* ``copy_mla_kv_with_rope_correction`` - the MLA analogue. MLA's pool
  already hands back ``k_nope``/``k_rope`` as separate tensors (no
  ``[..., :rotary_dim]`` slicing needed), and ``k_nope`` plays V's role
  too (MLA has no separate V buffer).

* ``as_long_tensor`` - coerce list/numpy/torch input to a long-typed
  torch.Tensor on a target device. Used for segment-pos plumbing where
  Python providers may produce plain lists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention


def as_long_tensor(obj, device) -> torch.Tensor:
    """Coerce list/numpy/torch input to a long-typed torch.Tensor on ``device``."""
    if obj is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=torch.long, non_blocking=True)
    return torch.as_tensor(list(obj), dtype=torch.long, device=device)


def _donor_target_cos_sin(cos_sin_cache, old_positions, new_positions):
    old_cos_sin = cos_sin_cache.index_select(0, old_positions)
    new_cos_sin = cos_sin_cache.index_select(0, new_positions)
    old_cos, old_sin = old_cos_sin.chunk(2, dim=-1)
    new_cos, new_sin = new_cos_sin.chunk(2, dim=-1)
    return old_cos, old_sin, new_cos, new_sin


def copy_kv_with_rope_correction(
    pool,
    rotary_emb,
    old_locs: torch.Tensor,
    new_locs: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    layer_recompute_mask: Optional[List[bool]] = None,
    *,
    apply_rotary_emb=None,
    reverse_rotary_emb=None,
) -> None:
    """Copy donor K/V into ``new_locs`` with RoPE re-indexed to ``new_positions``.

    For each layer:
        V[new_locs] = V[old_locs]                  (no position dependency)
        K[new_locs] = apply_rotary(reverse_rotary(K[old_locs], old_pos), new_pos)

    The K split into ``k_rot`` (the first ``rotary_dim`` channels, which
    carry RoPE) and ``k_pass`` (channels past ``rotary_dim``, which don't)
    matches SGLang's rotary embedding kernel layout. Both halves go back
    into K via ``torch.cat`` so the resulting tensor has the same shape
    as the original K.

    Args:
        pool: KV pool with ``k_buffer`` / ``v_buffer`` lists, ``layer_num``.
        rotary_emb: Rotary embedding object exposing ``cos_sin_cache``,
            ``is_neox_style``, ``rotary_dim``.
        old_locs: Source slot indices in the donor's KV pool.
        new_locs: Destination slot indices in the recipient's KV pool.
            Must have the same length as ``old_locs``.
        old_positions: Donor-side absolute positions for ``old_locs``.
            Used to invert RoPE from the donor's reference frame.
        new_positions: Target-side absolute positions for ``new_locs``.
            Used to apply RoPE in the recipient's reference frame.
        layer_recompute_mask: Optional list of bools; when ``mask[i]`` is
            True, layer ``i`` is zeroed instead of copied (the prefill
            pass that bookends the block will produce fresh K/V for those
            layers). Bathtub-curve drift mitigation. List shorter than
            ``pool.layer_num`` is treated as no-mask for trailing layers.
        apply_rotary_emb / reverse_rotary_emb: Optional overrides for the
            rotary embedding kernels. Default: lazy-imported from SGLang's
            ``layers.rotary_embedding.utils``. The override hook lets unit
            tests run without pulling in SGLang's heavyweight rotary stack
            (which uses ``@torch.compile``).
    """
    if apply_rotary_emb is None:
        from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb as _apply

        apply_rotary_emb = _apply
    if reverse_rotary_emb is None:
        from sglang.srt.layers.rotary_embedding.utils import (
            reverse_rotary_emb as _reverse,
        )

        reverse_rotary_emb = _reverse

    cos_sin_cache = rotary_emb.cos_sin_cache
    is_neox_style = rotary_emb.is_neox_style
    rotary_dim = rotary_emb.rotary_dim

    old_cos, old_sin, new_cos, new_sin = _donor_target_cos_sin(
        cos_sin_cache, old_positions, new_positions
    )

    mask_len = len(layer_recompute_mask) if layer_recompute_mask else 0
    for layer_id in range(pool.layer_num):
        if layer_id < mask_len and layer_recompute_mask[layer_id]:
            pool.v_buffer[layer_id][new_locs] = 0
            pool.k_buffer[layer_id][new_locs] = 0
            continue

        pool.v_buffer[layer_id][new_locs] = pool.v_buffer[layer_id][old_locs]

        k = pool.k_buffer[layer_id][old_locs]
        k_rot = k[..., :rotary_dim]
        k_pass = k[..., rotary_dim:]

        k_raw = reverse_rotary_emb(k_rot, old_cos, old_sin, is_neox_style)
        k_new = apply_rotary_emb(k_raw, new_cos, new_sin, is_neox_style)

        pool.k_buffer[layer_id][new_locs] = torch.cat((k_new, k_pass), dim=-1)


def copy_mla_kv_with_rope_correction(
    pool,
    attn_layers: List[RadixAttention],
    rotary_emb,
    old_locs: torch.Tensor,
    new_locs: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    layer_recompute_mask: Optional[List[bool]] = None,
    *,
    apply_rotary_emb=None,
    reverse_rotary_emb=None,
) -> None:
    """MLA analogue of ``copy_kv_with_rope_correction``.

    Conceptually, for each layer:
        k_nope[new_locs] = k_nope[old_locs]                     (no position dependency)
        k_rope[new_locs] = apply_rotary(reverse_rotary(k_rope[old_locs], old_pos), new_pos)

    but the rotation itself runs **once, batched across every active layer**,
    not once per layer. Profiling (see ``POC_PLAN.md``'s "Context-window
    sweep") found a per-layer Python loop spends ~95% of its wall-clock time
    on CPU-side dispatch/kernel-launch overhead between many small elementwise
    ops, not GPU compute — ``reverse_rotary_emb``/``apply_rotary_emb`` each
    issue ~9 tiny kernels (dtype casts, chunk, multiply, add/subtract, cat),
    repeated per layer. Since ``old_cos``/``old_sin``/``new_cos``/``new_sin``
    depend only on position, not layer, and both rotary helpers are already
    pure elementwise ops that broadcast over any leading batch dims, stacking
    every active layer's ``k_rope`` into one ``[num_tokens, num_active_layers,
    rope_dim]`` tensor and calling each rotary helper once collapses
    ``num_active_layers`` invocations of that ~9-kernel chain into one.
    ``k_nope`` still round-trips per layer via ``get_mla_kv_buffer``/
    ``set_mla_kv_buffer`` (each already a single fused Triton kernel, and MLA
    has no separate V buffer to batch it against), and masked (recompute)
    layers are excluded from the batch entirely and zeroed directly, same as
    before — only the rotation math itself is batched.

    Unlike the MHA path there's no partial-rotary split within one tensor:
    ``DeepseekV2AttentionMLA`` constructs its rotary embedding with
    ``rotary_dim == qk_rope_head_dim`` (the entire ``k_rope`` channel width
    is rotary), and ``MLATokenToKVPool.get_mla_kv_buffer``/
    ``set_mla_kv_buffer`` already hand back/accept ``k_nope``/``k_rope`` as
    separate tensors, so there's nothing to slice. ``k_nope`` also plays
    V's role here (MLA has no separate V buffer — see
    ``MLATokenToKVPool.get_value_buffer``).

    Args:
        pool: ``MLATokenToKVPool`` exposing ``get_mla_kv_buffer``/
            ``set_mla_kv_buffer``, ``kv_buffer`` (list, per layer),
            ``start_layer``.
        attn_layers: The ``RadixAttention`` instance per decoder layer, in
            the same order ``layer_recompute_mask`` (if given) is indexed
            by. This is ``model.model.layers[i].self_attn.attn_mqa`` —
            ``get_mla_kv_buffer``/``set_mla_kv_buffer`` read only
            ``.layer_id`` off it, but require the real object, not a raw
            int.
        rotary_emb: Same contract as ``copy_kv_with_rope_correction``.
        old_locs / new_locs / old_positions / new_positions: Same
            semantics as ``copy_kv_with_rope_correction``.
        layer_recompute_mask: Same semantics as
            ``copy_kv_with_rope_correction``. A flagged layer's combined
            latent buffer is zeroed directly (bit-pattern zero is
            dtype-correct regardless of the nope/rope packing), rather
            than going through the read/write accessors.
        apply_rotary_emb / reverse_rotary_emb: Same override hooks as
            ``copy_kv_with_rope_correction``, for CPU-only unit testing.
    """
    if apply_rotary_emb is None:
        from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb as _apply

        apply_rotary_emb = _apply
    if reverse_rotary_emb is None:
        from sglang.srt.layers.rotary_embedding.utils import (
            reverse_rotary_emb as _reverse,
        )

        reverse_rotary_emb = _reverse

    cos_sin_cache = rotary_emb.cos_sin_cache
    is_neox_style = rotary_emb.is_neox_style
    assert rotary_emb.rotary_dim == pool.qk_rope_head_dim, (
        "MLA correction assumes the entire k_rope slice is rotary "
        f"(rotary_dim={rotary_emb.rotary_dim}, "
        f"qk_rope_head_dim={pool.qk_rope_head_dim})"
    )

    old_cos, old_sin, new_cos, new_sin = _donor_target_cos_sin(
        cos_sin_cache, old_positions, new_positions
    )

    mask_len = len(layer_recompute_mask) if layer_recompute_mask else 0
    active_layers = []
    k_nopes = []
    k_ropes = []
    for i, attn_layer in enumerate(attn_layers):
        if i < mask_len and layer_recompute_mask[i]:
            pool.kv_buffer[attn_layer.layer_id - pool.start_layer][new_locs] = 0
            continue

        k_nope, k_rope = pool.get_mla_kv_buffer(attn_layer, old_locs)
        active_layers.append(attn_layer)
        k_nopes.append(k_nope)
        k_ropes.append(
            k_rope.squeeze(1)
        )  # [num_tokens, 1, rope_dim] -> [num_tokens, rope_dim]

    if not active_layers:
        return

    # [num_tokens, num_active_layers, rope_dim] -- cos/sin depend only on
    # position, so both rotary helpers broadcast over the layer dim unchanged.
    k_rope_stack = torch.stack(k_ropes, dim=1)
    k_raw_stack = reverse_rotary_emb(k_rope_stack, old_cos, old_sin, is_neox_style)
    k_rope_new_stack = apply_rotary_emb(k_raw_stack, new_cos, new_sin, is_neox_style)

    for i, attn_layer in enumerate(active_layers):
        pool.set_mla_kv_buffer(
            attn_layer, new_locs, k_nopes[i], k_rope_new_stack[:, i : i + 1, :]
        )
