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

"""Fused gather-rotate-scatter MLA realization kernel (tier (a), fuse_kernel.md S4.2).

Replaces the per-layer realization loop in
``copy_mla_kv_with_rope_correction`` -- 2 * num_layers Triton launches plus
2 * num_layers ``torch.empty`` allocations per fuzzy hit -- with a single
Triton launch over grid ``(num_layers, num_tokens)`` and zero intermediate
allocations. Profiling showed the unfused path is dispatch-bound (93-98% of
gather/scatter time at 700 donor tokens is not memory traffic), so the win
here is launch count, not bandwidth.

Per (layer, token) program:
  * k_nope / c_KV slice (row[0:NOPE_DIM]) is position-free: straight copy.
  * k_rope slice (row[NOPE_DIM:NOPE_DIM+2*ROPE_HALF]) is rotated from the
    donor's absolute position to the recipient's, in registers, as
    ``apply_rotary_emb(reverse_rotary_emb(k_rope, old), new)`` -- the exact
    op order of the reference helpers in ``layers/rotary_embedding/utils.py``,
    computed in fp32 and rounded once to bf16 at the store. Both rotary lane
    conventions are supported via the IS_NEOX constexpr: neox-style pairs
    lanes (r, r+HALF); gptj-style pairs lanes (2r, 2r+1) (the real
    DeepSeek-V2-Lite/Moonlight configs are gptj-style).
  * A ``layer_recompute_mask``-flagged layer is *zeroed* at the destination,
    not copied (matching ``rope_correction.py``'s masked-layer path); the
    mask is a kernel input indexed by layer id, so masked layers are part of
    the same single launch.

Scope (asserted in the wrapper, matching fuse_kernel.md's narrow scope):
single GPU, bf16, ``rotary_dim == qk_rope_head_dim`` (the entire k_rope
slice is rotary), pool ``start_layer == 0`` with one attn layer per pool
layer. FP8 / DCP / page_size>1 layouts are out of scope.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import torch

from sglang.srt.mem_cache.fuzzy_match.rope_correction import (
    _donor_target_cos_sin,
    as_long_tensor,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

import triton
import triton.language as tl


@triton.jit
def _fused_mla_rope_delta_kernel(
    data_ptrs,  # uint64 [num_layers]: per-layer kv_buffer base pointers
    layer_mask_ptr,  # uint8 [num_layers]: nonzero => zero the row, don't copy
    old_locs_ptr,  # int64 [n]: donor slot indices
    new_locs_ptr,  # int64 [n]: freshly-allocated recipient slot indices
    old_cos_ptr,  # bf16 [n, ROPE_HALF], contiguous
    old_sin_ptr,  # bf16 [n, ROPE_HALF], contiguous
    new_cos_ptr,  # bf16 [n, ROPE_HALF], contiguous
    new_sin_ptr,  # bf16 [n, ROPE_HALF], contiguous
    row_stride,  # elements per kv row (= NOPE_DIM + 2 * ROPE_HALF)
    NOPE_DIM: tl.constexpr,  # position-free prefix of the row (kv_lora_rank)
    ROPE_HALF: tl.constexpr,  # half the rotary slice (qk_rope_head_dim // 2)
    IS_NEOX: tl.constexpr,  # neox: lanes (r, r+HALF) pair; gptj: lanes (2r, 2r+1)
):
    bid = tl.program_id(0)  # pool layer index (== index into data_ptrs)
    tok = tl.program_id(1)  # token index into the loc arrays

    base = tl.load(data_ptrs + bid)
    base = tl.cast(base, tl.pointer_type(tl.bfloat16))

    new = tl.load(new_locs_ptr + tok).to(tl.int64)
    dst_row = base + new * row_stride

    if tl.load(layer_mask_ptr + bid) != 0:
        # Bathtub-curve recompute path: a flagged layer is zeroed outright
        # (bit-pattern zero is dtype-correct), never copied.
        tl.store(
            dst_row + tl.arange(0, NOPE_DIM),
            tl.zeros([NOPE_DIM], dtype=tl.bfloat16),
        )
        tl.store(
            dst_row + NOPE_DIM + tl.arange(0, 2 * ROPE_HALF),
            tl.zeros([2 * ROPE_HALF], dtype=tl.bfloat16),
        )
        return

    old = tl.load(old_locs_ptr + tok).to(tl.int64)
    src_row = base + old * row_stride

    # Position-free slice: plain copy.
    nope_offs = tl.arange(0, NOPE_DIM)
    tl.store(dst_row + nope_offs, tl.load(src_row + nope_offs))

    # Rotary slice, paired lanes: neox pairs (r, r+HALF); gptj pairs (2r, 2r+1).
    # One cos/sin value per pair, same math either way.
    r = tl.arange(0, ROPE_HALF)
    co = tl.load(old_cos_ptr + tok * ROPE_HALF + r).to(tl.float32)
    so = tl.load(old_sin_ptr + tok * ROPE_HALF + r).to(tl.float32)
    cn = tl.load(new_cos_ptr + tok * ROPE_HALF + r).to(tl.float32)
    sn = tl.load(new_sin_ptr + tok * ROPE_HALF + r).to(tl.float32)

    if IS_NEOX:
        o1_off = NOPE_DIM + r
        o2_off = NOPE_DIM + ROPE_HALF + r
    else:
        o1_off = NOPE_DIM + 2 * r
        o2_off = NOPE_DIM + 2 * r + 1

    o1 = tl.load(src_row + o1_off).to(tl.float32)
    o2 = tl.load(src_row + o2_off).to(tl.float32)

    # reverse_rotary_emb at the donor position (inverse of forward RoPE).
    x1 = o1 * co + o2 * so
    x2 = o2 * co - o1 * so
    # apply_rotary_emb at the recipient position.
    y1 = x1 * cn - x2 * sn
    y2 = x2 * cn + x1 * sn

    tl.store(dst_row + o1_off, y1.to(tl.bfloat16))
    tl.store(dst_row + o2_off, y2.to(tl.bfloat16))


def copy_mla_kv_with_rope_correction_fused(
    pool,
    attn_layers: List[RadixAttention],
    rotary_emb,
    old_locs: torch.Tensor,
    new_locs: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    layer_recompute_mask: Optional[List[bool]] = None,
) -> None:
    """Fused single-launch variant of ``copy_mla_kv_with_rope_correction``.

    Same contract as the reference: donor rows at ``old_locs`` / donor
    absolute ``old_positions`` are copied into the freshly-allocated
    ``new_locs`` with k_rope re-indexed to ``new_positions``;
    ``layer_recompute_mask``-flagged layers are zeroed at ``new_locs``.
    ``old_locs`` and ``new_locs`` must be disjoint (recipient slots are
    freshly allocated by ``match_prefix``), which the recipient-pre-alloc
    contract in ``realizer.py`` already guarantees; the kernel is therefore
    not safe for overlapping ranges, same as the reference accessors.

    Constraints beyond the reference's, all asserted (fail loud -- the fused
    path is an explicit opt-in via SGLANG_FUZZY_REALIZE_FUSED):
      * pool rows are bf16 (the kernel's pointer cast assumes it),
      * the entire k_rope slice is rotary
        (``rotary_dim == qk_rope_head_dim``),
      * ``attn_layers[i].layer_id == pool.start_layer + i`` for all i and
        ``pool.start_layer == 0`` (single-GPU 1:1 layer mapping).
    Neox-style and gptj-style rotary lane pairings are both supported.
    """
    device = pool.device
    num_layers = pool.layer_num

    assert pool.kv_buffer[0].dtype == torch.bfloat16, (
        "fused MLA realization is bf16-only "
        f"(kv_buffer dtype {pool.kv_buffer[0].dtype}); use the reference path"
    )
    assert rotary_emb.rotary_dim == pool.qk_rope_head_dim, (
        "fused MLA realization assumes the entire k_rope slice is rotary "
        f"(rotary_dim={rotary_emb.rotary_dim}, "
        f"qk_rope_head_dim={pool.qk_rope_head_dim})"
    )
    assert pool.start_layer == 0 and len(attn_layers) == num_layers, (
        "fused MLA realization assumes a single-GPU 1:1 layer mapping "
        f"(start_layer={pool.start_layer}, num_layers={num_layers}, "
        f"attn_layers={len(attn_layers)})"
    )
    assert pool.kv_buffer[0].is_contiguous()

    old_locs = as_long_tensor(old_locs, device).contiguous()
    new_locs = as_long_tensor(new_locs, device).contiguous()
    n = old_locs.numel()
    assert new_locs.numel() == n
    if n == 0:
        return
    old_positions = as_long_tensor(old_positions, device)
    new_positions = as_long_tensor(new_positions, device)

    old_cos, old_sin, new_cos, new_sin = _donor_target_cos_sin(
        rotary_emb.cos_sin_cache, old_positions, new_positions
    )
    # Downcast to the pool dtype exactly as apply_rotary_emb/reverse_rotary_emb
    # do (`.to(x.dtype)` with bf16 operands), and make each [n, HALF] block
    # contiguous for flat in-kernel indexing.
    old_cos = old_cos.to(torch.bfloat16).contiguous()
    old_sin = old_sin.to(torch.bfloat16).contiguous()
    new_cos = new_cos.to(torch.bfloat16).contiguous()
    new_sin = new_sin.to(torch.bfloat16).contiguous()

    mask = torch.zeros(num_layers, dtype=torch.uint8, device=device)
    if layer_recompute_mask:
        mask[: len(layer_recompute_mask)] = torch.as_tensor(
            layer_recompute_mask, dtype=torch.uint8, device=device
        )

    nope_dim = pool.kv_lora_rank
    rope_half = pool.qk_rope_head_dim // 2
    row_stride = pool.kv_buffer[0].stride(0)

    _fused_mla_rope_delta_kernel[(num_layers, n)](
        pool.data_ptrs,
        mask,
        old_locs,
        new_locs,
        old_cos,
        old_sin,
        new_cos,
        new_sin,
        row_stride,
        NOPE_DIM=nope_dim,
        ROPE_HALF=rope_half,
        IS_NEOX=rotary_emb.is_neox_style,
        num_warps=4,
    )
