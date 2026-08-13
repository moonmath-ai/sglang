# Position-Independent Caching (Irminsul PIC) for MLA models on sglang

Port of the single-GPU Irminsul proof-of-concept onto latest mainline sglang
(branch `feat/Irminsul`, base `fd1e04d95`). **Educational / research code, not
for production.** ExactHash-only: the SemBlend/`SemanticEmbedding` provider from
the original PR is intentionally not ported.

**TL;DR of what is proven (all measured, not claimed):**

- **Correctness: solid.** On this branch: avg KL **0.0030** (DeepSeek-V2-Lite-Chat)
  / **0.0019** (Moonlight-16B-A3B-Instruct) against a 0.02 gate — digit-for-digit
  parity with the reference PoC (0.0032 / 0.0019 on the same Blackwell GPU class).
  argmax match ≈ 0.99. Recovery 99.3–99.4% on the controlled probe.
- **Performance: conditional.** A PIC hit is faster than a recompute only above
  ~**800 donor tokens** (fused-kernel break-even). On *position-shifted reuse*
  traffic (a long retrieved doc reappearing mid-prompt: "RAG-shaped"), the
  original PoC measured **1.18–1.28x end-to-end**. On prefix-aligned agentic
  traffic plain RadixCache already recovers ~96%, and PIC breaks even (1.01x)
  — its ~3–5% always-on bookkeeping eats the gain. No free lunch: the win is
  workload-shaped, not universal. See §6 and §8.

## 1. The problem and the mechanism

Standard RadixAttention reuses KV only when the incoming prefix is
byte-identical **and at the same absolute offset** as cached content.
RAG/agentic traffic — retrieved documents, tool schemas, boilerplate — puts
identical content at *different* offsets across requests: guaranteed miss.

For **MLA** models the fix is cheap. The cached KV row splits into a
position-free latent slice (`k_nope`, `kv_lora_rank` dims — copied verbatim)
and a rotary slice (`k_rope`, `qk_rope_head_dim` dims). A cached `k_rope`
computed for position `p_src` can be re-indexed to `p_tgt` exactly by
reversing RoPE at `p_src` and applying it at `p_tgt`:

```
k_tgt = apply_rotary_emb(reverse_rotary_emb(k_src, cos/p_src, sin/p_src), cos/p_tgt, sin/p_tgt)
```

No recompute, no approximation beyond one bf16 rounding (rel-L2 ≈ 3e-3 in
bf16, matching the paper's 4.7e-3). MHA models have no position-free slice,
so for them this mechanism is pure position bookkeeping (validated as the
reference path, not the paper's claim).

## 2. Architecture

Module `python/sglang/srt/mem_cache/fuzzy_match/` (~2,300 lines; see its
`README.md` for the full call chain and lock/alloc ownership):

| File | Role |
|---|---|
| `chunker.py` | Content-defined chunking (CDC, Gear hash, ~128-token expected chunks, 32-token attention-sink carve-out). Boundary decisions depend only on the trailing 64 tokens — position-independent by construction. |
| `exact_hash_provider.py` | Provider: CDC + exact-hash donor index with mandatory per-chunk token-ID equality ("never trust the hash alone"); greedy same-donor contiguous-run extension (multi-chunk). |
| `fuzzy_radix_cache.py` | `FuzzyRadixCache(RadixCache)` + factory. Validates matches, pre-allocates realization slots, pins donor nodes (`inc_lock_ref`), registers finished requests as donors (default alignment + divergence-point alignment), gating (`fuzzy_min_suffix_tokens`, `fuzzy_min_match_length`). |
| `realizer.py` | `FuzzyKVRealizer`: pre-forward gather → RoPE delta-rotate → scatter of donor KV into recipient slots; repoints `req_to_token`; clears per-request state so chunked prefill/decode/retract never re-trigger. |
| `rope_correction.py` | Per-layer reference implementations (MHA + MLA, rotation batched across layers). The correctness oracle. |
| `fused_mla_realize.py` | Opt-in single-launch Triton gather-rotate-scatter kernel (`SGLANG_FUZZY_REALIZE_FUSED=1`), bf16, neox/gptj lanes. Cuts `_copy_kv` 12.9×. |
| `config.py` | `FuzzyMatchConfig` (ExactHash-only). |

Wiring seams (all inert when `--radix-cache-backend` is unset):
`MatchResult.fuzzy_matched_len` field; `RadixCache._on_finished_insert` no-op
hook; registry lazy import; 5 `Req` fields + match consumption +
`reset_for_retract` reset; `ForwardBatch.fuzzy_reqs`; `ModelRunner`
`maybe_init_fuzzy_kv_realizer` + a realize call in `_forward_raw` (after the
deferred-mamba hook, before extend dispatch — outside CUDA-graph capture);
`#fuzzy-token` prefill log; `--fuzzy-match-provider` / `--fuzzy-min-match-length`
server args (`NS("memory")`); 3 env vars; `MLATokenToKVPool` `data_strides` +
`enable_kv_cache_copy` + batched `move_kv_cache` (overlap-checked fallback);
`reverse_rotary_emb` in `layers/rotary_embedding/utils.py`.

## 3. Usage

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \
  --radix-cache-backend fuzzy_match          # the enable switch
  # --fuzzy-match-provider ExactHash         # default; only in-tree provider
  # --fuzzy-min-match-length 16              # default
# optional: SGLANG_FUZZY_REALIZE_FUSED=1 for the single-launch kernel
```

Markers in the server log: `[FUZZY RADIX] fuzzy match success`,
`[FUZZY] Realized N fuzzy tokens`, `#fuzzy-token: N` in the prefill line.

Tests:

```bash
# CPU unit (41 tests)
pytest test/registered/unit/mem_cache/ -k "fuzzy or chunker" -q
# CUDA unit (fused-kernel parity)
pytest test/registered/unit/mem_cache/fuzzy_match/test_fused_mla_realize.py -v
# E2E (self-launching servers; shared-GPU-friendly resource envelope baked in)
pytest test/registered/fuzzy_match/ -s
# Measurement harness (TTFT sweep, RAG/BFCL replays, profiler): test/manual/fuzzy_match/
```

## 4. Validation of this port (this branch, this GPU, 2026-08-13)

| Gate | Result |
|---|---|
| CPU unit (41) | pass |
| CUDA unit fused parity (5) | pass (fused ≡ reference ≡ float64 oracle) |
| E2E KL, DeepSeek-V2-Lite (T2) | avg KL **0.0030** (PoC ref 0.0032), argmax 0.9922, `[FUZZY] Realized` 8/8 |
| E2E KL, Moonlight-16B-A3B (T3) | avg KL **0.0019** (ref 0.0019), multi-chunk 0.0013 |
| E2E KL, Qwen2.5-7B-AWQ (MHA) | avg KL **0.0103** (< 0.02 gate; above the old H100 figure 0.0041 — tracked) |
| E2E safety | forced-collision disambiguation 8/8; cross-tenant `extra_key` isolation pass |
| Server boot, default backend | unchanged behavior, byte-identical output |

## 5a. This branch — measured TTFT (feat/Irminsul, RTX PRO 6000, 2026-08-13)

Mean wall-clock ms per request (`max_new_tokens=1`, isolated single request,
`--attention-backend flashinfer`, full GPU). HIT = cached donor reused at a
shifted position (realize + prefill the remainder); MISS = same-size span
never registered (full recompute). Ratio = MISS ÷ HIT, >1 means reuse wins.
Each cell is the mean of 10 measured repeats; `[FUZZY] Realized` fired on all
65 HIT trials per arm.

| donor tokens | unfused HIT | unfused MISS (ratio) | fused HIT | fused MISS (ratio) |
|---|---|---|---|---|
| 100 | — | — | 34.3 | 33.8 (0.98x) |
| 200 | — | — | 38.8 | 39.3 (1.01x) |
| 300 | — | — | 29.8 | 39.3 (1.32x) |
| 400 | — | — | 34.4 | 41.3 (1.20x) |
| 500 | — | — | 34.6 | 42.6 (1.23x) |
| 700 | 40.6 | 47.8 (1.18x) | 35.8 | 46.1 (1.29x) |
| 1000 | 37.5 | 52.7 (1.41x) | 35.0 | 52.6 (1.50x) |
| 1500 | 40.3 | 63.2 (1.57x) | 37.3 | 62.9 (1.68x) |
| 3000 | 42.1 | 108.8 (2.58x) | 40.0 | 108.8 (2.72x) |
| 6000 | 47.1 | 235.0 (4.99x) | 44.8 | 234.9 (5.24x) |

**The HIT column is the one that moves least**: realization cost is set by
`num_hidden_layers`, not token count (fused: 34.3 → 44.8ms across 100 → 6000
donor tokens), while MISS grows superlinearly with prompt size (attention is
quadratic: 33.8 → 234.9ms). Ratios alone hide this shape — the reuse reshape
is "pay a small flat tax instead of a growing one", not "PIC gets faster with
size". Break-even on this branch is **≈ 200 donor tokens fused** (1.01x at
200, 0.98x at 100) — versus ≈ 800 in the original PoC (see 5b).

**The RAG arm reproduces on this branch (same box, 2026-08-13, sequential
replay of 24 conversations × 4 turns, 1500-token wikitext docs at rotated
offsets, append-only transcripts):** plain RadixCache 7.83s / 67.5% recovery
vs fuzzy_match 6.71s / 80.4% recovery — **1.17x end-to-end** (PoC: 1.18x,
same +12.8pp recovery delta), PIC faster on 53% of paired requests with the
always-on overhead included. Fuzzy bookkeeping measured at 0.3s total across
the 96-request run (match+registration), so on this traffic shape the per-hit
win is what you get. *Caution when measuring on a shared box: two earlier
contaminated runs showed the fuzzy arm at 12.4-12.6s (0.67x) with identical
recovery — same code, same workload, 2x wall-time swing from an unrelated GPU
tenant. A single slow reading on shared hardware means nothing.*

## 5b. Original PoC reference (July-2026 base, same GPU class)

Same methodology. *Reproducing these on this branch gives markedly better
HIT times — the HIT path's surrounding infra (scheduler/allocation/attention)
got faster upstream; MISS times are unchanged to within run variance.*

| donor tokens | unfused HIT | unfused MISS (ratio) | fused HIT | fused MISS (ratio) |
|---|---|---|---|---|
| 700 | 54.6 | 50.5 (0.92x) | 50.4 | 48.9 (0.97x) |
| 1000 | 54.9 | 53.9 (0.98x) | 50.6 | 53.9 (1.07x) |
| 1500 | 55.0 | 62.8 (1.14x) | 49.7 | 64.0 (1.29x) |
| 3000 | 56.9 | 105.2 (1.85x) | 52.8 | 107.7 (2.04x) |
| 6000 | 61.8 | 230.1 (3.72x) | 62.3 | 234.8 (3.77x) |

PoC-era break-even was ≈ 1060 tokens unfused, ≈ 800 fused. Note the fused
kernel still helps (+5-12% HIT at every size), but the dominant delta
branch-vs-PoC is the surrounding HIT-path plumbing, not the copy kernel.

Workload-shaped end-to-end (PoC, DeepSeek-V2-Lite):

- **RAG-shaped traffic: 1.17-1.18x** (this branch: 6.71s vs 7.83s, recovery
  80.4% vs 67.5%; PoC: 6.98s vs 8.24s, same recovery delta; 1.19x with fused) —
  1.19x with fused; survives concurrency 1.08–1.17x at c=1…24).
- **BFCL v3 agentic replay: 1.01x** (95.9% of reuse is already prefix-aligned
  and covered by plain RadixCache; PIC fires and is 0.94x on that subset).
- **Binding constraint: ~3–5% always-on bookkeeping** of the fuzzy path, not
  per-hit realization cost (fused kernel moved break-even only to ~800).

## 6. Known bugs & constraints

- **FP8 KV cache breaks the correction** (confirmed): the rotary helpers cast
  cos/sin to the 8-bit storage dtype. Needs an fp32 upcast of the rotary slice
  (never attempted). Do not combine PIC with quantized KV cache until fixed.
- **ExactHashProvider reuses only blocks starting at the recipient's first
  unmatched token** (prefix-shaped `device_indices` contract; cross-donor N:M
  segments exist as infra but are unexercised). Divergence-aligned donor
  registration mitigates; ~54/90 potential hits landed in the RAG arm.
- Rejected loudly at startup: hybrid SSM/Mamba models, EAGLE, hierarchical
  cache; unsupported KV pools and FP8 assert at use. TP > 1 untested.
- Always-on overhead (~3–5%) makes the fuzzy path counterproductive on
  prefix-aligned traffic (see roadmap item 1).

## 7. Performance roadmap (ordered by expected ROI)

1. **Conditional fuzzy path** — gate the provider lookup on expected gain
   (exact-prefix coverage low AND suffix above a runtime break-even model).
   Removes the always-on overhead where PIC has no headroom; the single
   biggest lever (repeatedly confirmed by measurement).
2. **Multi-donor segmented matching** — let `ExactHashProvider` emit the
   existing `FuzzyMatchSegment` format (best-k non-overlapping runs) instead of
   one contiguous span; attacks the 54/90 alignment bound.
3. **Hide realization behind the forward** — realize on a side CUDA stream at
   batch assembly, or layer-by-layer just ahead of each attention block;
   makes HIT TTFT ≈ MISS-of-suffix alone, dropping break-even well under 800.
4. **Multi-alignment donor registration** (sink + divergence + page) — cheap
   fingerprint-space multiplier for hit rate.
5. **FP8 upcast fix** in the fused kernel — unblocks quantized-KV deployments.
6. **TP > 1 correctness**, then GLM 5.2 bring-up (78 layers; realization cost
   scales ×2.9, fused kernel + tier-(b) FlashMLA read-time fusion lever).

## 8. Provenance

Ported from the private PoC (`POC_IRminsul-fuse-kernel` branch): PR #31057
(fuzzy-match radix backend) + 49 PIC commits (ExactHash provider, MLA
validation, multi-chunk, divergence-aligned registration, rotation batching,
fused kernel, measurement battery). Source docs include `IRMINSUL_THEORY.md`
(mechanism), `stage2_reproduction.md` (full measurement record), and
`fuse_kernel.md` (kernel design/build record). Paper: Irminsul, arXiv:2605.05696.
