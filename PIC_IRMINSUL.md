# Position-Independent Caching (Irminsul PIC) for MLA models on sglang

Implementation of the Irminsul mechanism (arXiv:2605.05696) — position-independent
KV-cache reuse via RoPE delta-rotation — as a pluggable radix-cache backend for
MLA models on mainline sglang (branch `feat/Irminsul`, base `fd1e04d95`).
**Educational / research code, not for production.** The in-tree provider is
`ExactHash` (lossless); the provider ABC admits out-of-tree providers.
All numbers below were measured on this branch (RTX PRO 6000 Blackwell,
2026-08-13) unless explicitly stated otherwise.

**TL;DR**

- **Correctness:** avg KL **0.0030** (DeepSeek-V2-Lite-Chat) / **0.0019**
  (Moonlight-16B-A3B-Instruct) against a 0.02 KL gate; argmax match ≈ 0.99.
- **Performance: conditional.** A PIC hit beats recompute above ~**200 donor
  tokens** (fused-kernel path; flat-HIT vs quadratic-MISS shape, §5). On
  *position-shifted reuse* traffic (a long retrieved doc reappearing mid-prompt:
  "RAG-shaped"), the win is **1.17x end-to-end** with overhead included. On
  prefix-aligned agentic traffic plain RadixCache already recovers ~96% and PIC
  breaks even (~1.0x) — there is nothing left for it to win. No free lunch: the
  win is workload-shaped, not universal. See §5 and §7.

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

No recompute, no approximation beyond one bf16 rounding. MHA models have no
position-free slice, so for them this mechanism is pure position bookkeeping —
validated end-to-end as a correctness reference, not as the Irminsul claim
itself (which is MLA-specific).

### How a match is found (the ExactHash matcher)

Exact prefix matching is always tried first and always wins; the fuzzy matcher
only runs on the unmatched tail. Per request:

1. **Chunk the tail position-independently.** The tail is split by
   content-defined chunking (CDC): a Gear-hash rolling window over the last 64
   tokens declares a boundary where its low 7 bits are zero (~128-token average
   chunks, content-defined, clamped 32–512). Because boundaries depend on
   content rather than absolute position, the same document produces the same
   chunks wherever it sits — the property the radix tree's chained,
   position-keyed hash lacks.
2. **Look up donors by fingerprint.** Each finished request registers its own
   chunks in an in-memory index keyed by `(extra_key, 64-bit exact-hash
   fingerprint)`, twice: once chunked from the attention-sink boundary and once
   from the request's own divergence point (so a recipient behind a different
   preamble can still line up). The first 32 tokens are never donors (the
   attention sink is position-driven, not content-stable).
3. **Verify, then extend.** A fingerprint hit is never trusted alone — the
   candidate's token IDs must be byte-equal. Matches extend greedily through
   consecutive chunks of the tail, but only while they keep coming from the
   *same donor at contiguous donor positions*: the result is one contiguous
   span in both token-ID and donor-position space, which is what the
   prefix-shaped scheduler contract (and the realization step below) can
   represent. Two cheap gates bound the no-hit cost: the tail must be at least
   `fuzzy_min_suffix_tokens` (256) long, and provider matches land only where a
   strong exact anchor doesn't already own the region.
4. **Pin and slot it in.** The donor's tree node is lock-pinned so eviction
   can't free its KV mid-forward, fresh recipient slots are pre-allocated, and
   the match is returned as `exact ++ fuzzy ++ remainder`. Before the forward
   pass the realizer applies the §1 delta-rotation from the donor's positions
   to the recipient's and repoints the request's KV map at the corrected
   slots.

## 2. Architecture

Module `python/sglang/srt/mem_cache/fuzzy_match/` (~2,300 lines; see its
`README.md` for the full call chain and lock/alloc ownership):

| File | Role |
|---|---|
| `chunker.py` | Content-defined chunking (CDC, Gear hash, ~128-token expected chunks, 32-token attention-sink carve-out). Boundary decisions depend only on the trailing 64 tokens — position-independent by construction. |
| `exact_hash_provider.py` | Provider: CDC + exact-hash donor index with mandatory per-chunk token-ID equality ("never trust the hash alone"); greedy same-donor contiguous-run extension (multi-chunk). |
| `fuzzy_radix_cache.py` | `FuzzyRadixCache(RadixCache)` + factory. Validates matches, pre-allocates realization slots, pins donor nodes (`inc_lock_ref`), registers finished requests as donors (default alignment + divergence-point alignment), gating (`fuzzy_min_suffix_tokens`, `fuzzy_min_match_length`). |
| `realizer.py` | `FuzzyKVRealizer`: pre-forward gather → RoPE delta-rotate → scatter of donor KV into recipient slots; repoints `req_to_token`; clears per-request state so chunked prefill/decode/retract never re-trigger. |
| `rope_correction.py` | Per-layer reference implementations (MHA + MLA, rotation batched across layers). The correctness oracle the fused kernel is checked against. |
| `fused_mla_realize.py` | Opt-in single-launch Triton gather-rotate-scatter kernel (`SGLANG_FUZZY_REALIZE_FUSED=1`), bf16, neox/gptj lanes. Collapses the per-layer realizer loop into one launch. |
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

## 4. Validation (2026-08-13, this branch)

| Gate | Result |
|---|---|
| CPU unit (41) | pass |
| CUDA unit fused parity (5) | pass (fused ≡ per-layer reference ≡ float64 oracle) |
| E2E KL, DeepSeek-V2-Lite (MLA) | avg KL **0.0030** (gate 0.02), argmax match 0.9922, `[FUZZY] Realized` 8/8; multi-chunk span (>512 tokens): KL 0.0067, argmax 0.9961 |
| E2E KL, Moonlight-16B-A3B (MLA) | avg KL **0.0019**; multi-chunk: KL 0.0013, argmax 0.9922 |
| E2E KL, Qwen2.5-7B-AWQ (MHA) | avg KL **0.0103** — passes the gate, the loosest arm by far; tracked |
| E2E safety | forced-collision disambiguation 8/8; cross-tenant `extra_key` isolation pass |
| Server boot, default backend | unchanged behavior, byte-identical output |

## 5. Measured performance (this branch)

### 5a. Per-hit TTFT donor sweep

Mean wall-clock ms per request (`max_new_tokens=1`, isolated single request,
`--attention-backend flashinfer`, full GPU, `measure_conditional_ttft_mla.py`).
HIT = cached donor reused at a shifted position (realize + prefill the
remainder); MISS = same-size span never registered (full recompute). Ratio =
MISS ÷ HIT, >1 means reuse wins. Each cell is the mean of 10 measured repeats;
`[FUZZY] Realized` fired on all HIT trials.

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
size". Break-even is **≈ 200 donor tokens fused** (1.01x at 200, 0.98x at
100).

### 5b. Workload-shaped traffic (A/B replay against the same server streams)

**RAG-shaped traffic — PIC's intended win: 1.17x end-to-end** (6.71s vs 7.83s
plain, recovery 80.4% vs 67.5% = +12.8pp, PIC faster on 53% of paired
requests; mean latency 69.9ms vs 81.6ms, p50 65.7ms vs 88.0ms; extra cached
tokens per request mean 499 / max 1523). Sequential replay of 24 conversations
× 4 turns, 1500-token wikitext docs reappearing at rotated absolute offsets,
append-only transcripts. Fuzzy bookkeeping (match + donor registration)
measured at 0.3s total across the 96-request run (~0.5ms per match call, ~2ms
per registration) — on this traffic shape the per-hit win is what you get.

**Agentic BFCL traffic — nothing to win: ~1.0x.** Single-tenant replay (130
paired requests): plain RadixCache already recovers 95.9% and PIC adds +0.00pp
and 1.01x end-to-end (42.0ms vs 42.3ms mean) — PIC fires on **zero** requests
(tool schemas sit at offset 0 and every turn is prefix-aligned). Multi-tenant
arm (4 tenants, variable preambles, 130 paired requests): PIC fires (mean
+86 / max +3072 extra cached tokens per request) and lands at 0.97x (65.0ms
vs 63.0ms mean) — the mechanism works but the traffic barely pays. Always-on
overhead on this branch: ~1–3% (unfused default; bookkeeping, not
realization).

**Why the fused kernel exists** (`profile_mla_realization_breakdown.py`, this
branch): at 700 donor tokens the unfused gather/scatter path runs at ~4–6% of
the 1513 GB/s peak — ~95% of it is kernel-launch dispatch, not memory traffic
(~1.4ms wall for 21.8MB of payload); at 4096 tokens it reaches ~33–40% of
peak. The fused single launch removes the dispatch floor at small donor
sizes, which is where the TTFT table's fused-vs-unfused gap comes from.

## 6. Known bugs & constraints

- **FP8 KV cache breaks the correction** (confirmed): the rotary helpers cast
  cos/sin to the 8-bit storage dtype. Needs an fp32 upcast of the rotary slice
  inside the rotate (never attempted). Do not combine PIC with quantized KV
  cache until fixed.
- **ExactHashProvider reuses only blocks starting at the recipient's first
  unmatched token** (prefix-shaped `device_indices` contract; cross-donor N:M
  segments exist as infra but are unexercised). Divergence-aligned donor
  registration mitigates but does not eliminate the alignment coincidence bound
  (60 of 96 replay requests realized in §5b's RAG arm).
- Rejected loudly at startup: hybrid SSM/Mamba models, EAGLE, hierarchical
  cache; unsupported KV pools and FP8 assert at use. TP > 1 untested.
- Always-on overhead (~1–3%) makes the fuzzy path counterproductive on
  prefix-aligned traffic (see roadmap item 1).

## 7. Performance roadmap (ordered by expected ROI)

1. **Conditional fuzzy path** — gate the provider lookup on expected gain
   (exact-prefix coverage low AND suffix above a runtime break-even model).
   Removes the always-on overhead where PIC has no headroom (the BFCL
   profile); the single biggest lever.
2. **Multi-donor segmented matching** — let `ExactHashProvider` emit the
   existing `FuzzyMatchSegment` format (best-k non-overlapping runs) instead of
   one contiguous span; attacks the alignment-bound hit ceiling seen in §5b.
3. **Hide realization behind the forward** — realize on a side CUDA stream at
   batch assembly, or layer-by-layer just ahead of each attention block;
   makes HIT TTFT ≈ MISS-of-suffix alone, driving break-even toward zero.
4. **Multi-alignment donor registration** (sink + divergence + page) — cheap
   fingerprint-space multiplier for hit rate.
5. **FP8 upcast fix** in the fused kernel — unblocks quantized-KV deployments.
6. **TP > 1 correctness**, then GLM 5.2 bring-up (78 layers; realization cost
   scales ×2.9 vs the 27-layer models tested here — the fused kernel becomes
   load-bearing, and FlashMLA read-side fusion is the bigger follow-up).
