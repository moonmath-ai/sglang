"""Profile the MLA realization path into its three phases, to decide whether a
fused gather/scatter kernel is worth building.

`POC_PLAN.md`'s "Potential way forward: a fused kernel" says explicitly that
before committing to the build, the gather/scatter cost has to be split into
"dispatch overhead" (fixable by batching across layers) vs. "genuine
memory-bandwidth-bound copy" (not fixable that way). This script measures that
split directly on real pool geometry, with no server and no model weights.

What it measures, per phase of `copy_mla_kv_with_rope_correction`:

  1. gather   -- `num_layers` x `get_mla_kv_buffer` (one Triton launch each)
  2. rotate   -- the already-batched rotation math (one call, all layers)
  3. scatter  -- `num_layers` x `set_mla_kv_buffer` (one Triton launch each)

and, for phases 1 and 3, compares wall-clock against two references:

  * **achieved bandwidth** vs. the device's measured peak (a large `copy_`),
    which says how much of the time is real memory traffic; and
  * **an empty-launch floor** -- the same number of launches of a trivial
    kernel, which says how much is pure CPU-side dispatch.

If dispatch dominates, batching across layers (tier (a) in POC_PLAN.md, using
the `data_ptrs` array `MLATokenToKVPool` already builds) recovers most of it.
If bandwidth dominates, it does not, and only tier (b) -- read-time fusion into
the attention kernel, which avoids the copy entirely -- can help.

Usage:
    python test/manual/fuzzy_match/profile_mla_realization_breakdown.py [num_tokens ...]

Defaults to the sizes the TTFT sweep uses (150, 512, 700) plus 4096.
"""

from __future__ import annotations

import sys

import torch

from sglang.srt.mem_cache.fuzzy_match.rope_correction import (
    copy_mla_kv_with_rope_correction,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

# DeepSeek-V2-Lite-Chat / Moonlight-16B-A3B geometry (both have 27 layers --
# confirmed from their real config.json, see POC_PLAN.md).
NUM_LAYERS = 27
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
POOL_SIZE = 65536
DTYPE = torch.bfloat16
MAX_POS = 8192

REPEATS = 20
WARMUP = 5


class _FakeLayer:
    """`get_mla_kv_buffer`/`set_mla_kv_buffer` read only `.layer_id`."""

    def __init__(self, layer_id: int):
        self.layer_id = layer_id


class _FakeRotary:
    """`copy_mla_kv_with_rope_correction` reads only these three attributes."""

    def __init__(self, rotary_dim: int, device: str):
        self.rotary_dim = rotary_dim
        self.is_neox_style = True
        # Same layout as SGLang's real rotary: [cos | sin] concatenated.
        inv_freq = 1.0 / (
            10000.0
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        t = torch.arange(MAX_POS, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(
            device=device, dtype=DTYPE
        )


def _time_cuda(fn, repeats=REPEATS, warmup=WARMUP) -> float:
    """Median wall-clock ms of `fn`, GPU-synchronized."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _measure_peak_bandwidth(device: str) -> float:
    """Measured device copy bandwidth in GB/s (read+write counted)."""
    n = 256 * 1024 * 1024 // 2  # 256 MiB of bf16
    src = torch.empty(n, dtype=DTYPE, device=device)
    dst = torch.empty(n, dtype=DTYPE, device=device)
    ms = _time_cuda(lambda: dst.copy_(src))
    moved_gb = (src.numel() * src.element_size() * 2) / 1e9
    return moved_gb / (ms / 1e3)


def _measure_launch_floor(device: str, num_launches: int) -> float:
    """ms for `num_launches` launches of a trivial kernel -- the dispatch floor."""
    tiny = torch.zeros(1, device=device)

    def do():
        for _ in range(num_launches):
            tiny.add_(1.0)

    return _time_cuda(do)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = "cuda"
    sizes = [int(a) for a in sys.argv[1:]] or [150, 512, 700, 4096]

    pool = MLATokenToKVPool(
        size=POOL_SIZE,
        page_size=1,
        dtype=DTYPE,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=NUM_LAYERS,
        device=device,
        enable_memory_saver=False,
    )
    layers = [_FakeLayer(i) for i in range(NUM_LAYERS)]
    rotary = _FakeRotary(QK_ROPE_HEAD_DIM, device)

    row_bytes = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * torch.tensor(
        [], dtype=DTYPE
    ).element_size()
    peak_bw = _measure_peak_bandwidth(device)
    print(f"device            : {torch.cuda.get_device_name(0)}")
    print(f"measured peak BW  : {peak_bw:.0f} GB/s (256MiB device copy, r+w)")
    print(
        f"pool geometry     : {NUM_LAYERS} layers x {KV_LORA_RANK}+{QK_ROPE_HEAD_DIM} dims, {DTYPE}"
    )
    print(f"row size          : {row_bytes} B/token/layer\n")

    for num_tokens in sizes:
        g = torch.Generator(device="cpu").manual_seed(0)
        old_locs = torch.randperm(POOL_SIZE, generator=g)[:num_tokens].to(device)
        new_locs = torch.randperm(POOL_SIZE, generator=g)[:num_tokens].to(device)
        old_pos = torch.randint(0, MAX_POS // 2, (num_tokens,), generator=g).to(device)
        new_pos = old_pos + 137

        # --- whole path, as the realizer actually calls it ---
        total_ms = _time_cuda(
            lambda: copy_mla_kv_with_rope_correction(
                pool, layers, rotary, old_locs, new_locs, old_pos, new_pos
            )
        )

        # --- phase 1: gather (num_layers launches) ---
        def gather():
            return [pool.get_mla_kv_buffer(l, old_locs) for l in layers]

        gather_ms = _time_cuda(gather)

        # --- phase 2: batched rotation math (1 call for all layers) ---
        bufs = gather()
        k_ropes = [r.squeeze(1) for _, r in bufs]
        k_nopes = [n for n, _ in bufs]
        from sglang.srt.layers.rotary_embedding.utils import (
            apply_rotary_emb,
            reverse_rotary_emb,
        )

        cs = rotary.cos_sin_cache
        oc, os_ = cs.index_select(0, old_pos).chunk(2, dim=-1)
        nc, ns = cs.index_select(0, new_pos).chunk(2, dim=-1)

        def rotate():
            st = torch.stack(k_ropes, dim=1)
            raw = reverse_rotary_emb(st, oc, os_, True)
            return apply_rotary_emb(raw, nc, ns, True)

        rotate_ms = _time_cuda(rotate)
        rotated = rotate()

        # --- phase 3: scatter (num_layers launches) ---
        def scatter():
            for i, l in enumerate(layers):
                pool.set_mla_kv_buffer(
                    l, new_locs, k_nopes[i], rotated[:, i : i + 1, :]
                )

        scatter_ms = _time_cuda(scatter)

        # --- references ---
        # gather reads num_layers*num_tokens rows and writes them out again;
        # scatter does the same. So each phase moves 2x the payload.
        payload_gb = (NUM_LAYERS * num_tokens * row_bytes) / 1e9
        moved_gb = payload_gb * 2
        bw_bound_ms = (moved_gb / peak_bw) * 1e3
        launch_floor_ms = _measure_launch_floor(device, NUM_LAYERS)

        print(
            f"=== num_tokens = {num_tokens} "
            f"({payload_gb*1e3:.2f} MB payload/phase) ==="
        )
        print(f"  full path             : {total_ms:8.3f} ms")
        print(f"  phase 1 gather  (x{NUM_LAYERS}) : {gather_ms:8.3f} ms")
        print(f"  phase 2 rotate  (x1)  : {rotate_ms:8.3f} ms")
        print(f"  phase 3 scatter (x{NUM_LAYERS}) : {scatter_ms:8.3f} ms")
        print(f"  -- references --")
        print(
            f"  bandwidth bound/phase : {bw_bound_ms:8.3f} ms  "
            f"(at {peak_bw:.0f} GB/s)"
        )
        print(
            f"  {NUM_LAYERS} empty launches    : {launch_floor_ms:8.3f} ms  "
            f"(pure dispatch floor)"
        )
        for name, ms in (("gather", gather_ms), ("scatter", scatter_ms)):
            frac_bw = bw_bound_ms / ms * 100
            print(
                f"  {name:7s}: {frac_bw:5.1f}% of it is bandwidth-bound, "
                f"so <= {100-frac_bw:5.1f}% is recoverable by batching "
                f"(achieved {moved_gb/(ms/1e3):.0f} GB/s = "
                f"{moved_gb/(ms/1e3)/peak_bw*100:.1f}% of peak)"
            )
        print()


if __name__ == "__main__":
    main()
