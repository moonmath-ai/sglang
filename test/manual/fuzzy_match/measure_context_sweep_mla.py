"""Context-window sweep: TTFT and TPOT (with percentiles) as the surrounding request
size grows, holding the PIC-realized span fixed at a single-chunk-friendly 150 tokens.

Motivation: reported externally that fuzzy-match/PIC benefit is "only realized past 8k
context windows." This POC's own conditional-TTFT measurement (fixed small total
context, ~350 tokens) found no measurable TTFT speedup, diagnosed as fixed
per-request overhead dominating at that scale (`POC_PLAN.md` Stage 2). This sweep
tests that diagnosis directly: does the same small, fixed PIC savings become
measurable once the surrounding request is large enough that fixed overhead stops
dominating?

**Important caveat, confirmed by reading the code, not assumed:** ``ExactHashProvider``
only ever realizes the *first* content-defined chunk of a matched donor
(`exact_hash_provider.py`'s `match_on_prefix_miss` -- literally `chunks[0]`, clamped to
`[32, 512]` tokens by `chunker.py`'s `MIN_CHUNK_TOKENS`/`MAX_CHUNK_TOKENS`), regardless
of how large the donor itself is, and `match_prefix` is called exactly once per request
at admission (not re-invoked mid-request as chunked prefill proceeds). This means
growing `DONOR_TOKENS` would NOT increase what actually gets PIC-realized -- it would
only grow the genuine-miss remainder after the capped first chunk, which would if
anything shrink the *relative* benefit as donor size grows past ~512 tokens. So this
sweep grows the *surrounding* exact-matched context (`query_prefix_tokens`) instead,
holding `DONOR_TOKENS` fixed at the same 150 tokens used throughout this POC's other
conditional-TTFT measurements. Testing whether a *larger reused span itself* shows more
benefit would require multi-chunk/segmented realization, which doesn't exist in this
codebase yet (`exact_hash_provider.py`'s own docstring: "Stage 1: single-chunk,
non-segmented matches only... Multi-chunk / N:M segment matches are a stretch goal") --
out of scope here, a separate follow-up.

TTFT measured via `meta_info["e2e_latency"]` at `max_new_tokens=1` -- server-side
latency, no HTTP-transport noise, unlike a client-side wall-clock timer (confirmed via
a live smoke test that this field is always populated, no `--enable-metrics` flag
needed). TPOT (decode-phase per-token latency) via a paired `max_new_tokens=N` call
under identical registration/content (same rng seed reused for both sub-calls),
computed as `(e2e_latency_N - e2e_latency_1) / (N - 1)`. TPOT is expected to be *flat*
across HIT/MISS at a given context size -- PIC only touches prefill; decode reads the
KV pool identically either way regardless of how it got populated. This is a
no-regression check, not a benefit metric.

Reports mean/P50/P75/P90/P95 for both metrics -- `MEASURED_REPEATS` raised to 20 (vs.
this repo's usual 10 for other manual scripts) since P95 needs more samples to mean
anything at all.

Not a CI test; run manually against an already-running fuzzy_match+ExactHash server:

    sglang serve --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \\
      --radix-cache-backend fuzzy_match --fuzzy-match-provider ExactHash \\
      --host 127.0.0.1 --port 21000
    python test/manual/fuzzy_match/measure_context_sweep_mla.py http://127.0.0.1:21000
"""

import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))

from sglang.test.kl_test_utils import _flush_cache, _generate

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:21000"

SINK_TOKENS = 32  # must match chunker.SINK_TOKENS
DONOR_TOKENS = 150  # single-chunk-friendly, matches measure_conditional_ttft_mla.py
DECODE_N = 32  # tokens generated for the TPOT sub-call
WARMUP_REPEATS = 3
MEASURED_REPEATS = 20
SEED = 20260802
LOW, HIGH = 1000, 99000  # DeepSeek-V2-Lite-Chat vocab_size=102400

# Total context sizes to sweep (query_prefix + donor); spans below/at/above the
# reported 8k threshold. DeepSeek-V2-Lite's YaRN scaling supports up to 163840, so
# the full range is reachable (unlike Moonlight, capped at max_position_embeddings
# =8192 -- see the _moonlight variant of this script for its narrower sweep).
TOTAL_CONTEXT_SIZES = [200, 2000, 8000, 16000, 32000]


def rand_tokens(rng, n):
    return [rng.randint(LOW, HIGH) for _ in range(n)]


def run_condition(condition, query_prefix_tokens, max_new_tokens, seed):
    """One request under ``condition`` ("hit" or "miss"); returns
    ``meta_info["e2e_latency"]`` (server-side, seconds)."""
    rng = random.Random(seed)
    donor_content = rand_tokens(rng, DONOR_TOKENS)
    query_prefix = rand_tokens(rng, query_prefix_tokens)
    if condition == "miss":
        tail = rand_tokens(rng, DONOR_TOKENS)  # never registered -> genuine miss
    else:
        tail = donor_content  # byte-identical to the registered donor -> PIC hit

    _flush_cache(BASE_URL)
    sink_prefix = list(range(50000, 50000 + SINK_TOKENS))
    _generate(BASE_URL, [sink_prefix + donor_content], max_new_tokens=0)
    _generate(BASE_URL, [query_prefix], max_new_tokens=0)

    result = _generate(BASE_URL, [query_prefix + tail], max_new_tokens=max_new_tokens)[
        0
    ]
    return result["meta_info"]["e2e_latency"]


def one_trial(condition, query_prefix_tokens, seed):
    ttft = run_condition(condition, query_prefix_tokens, max_new_tokens=1, seed=seed)
    e2e_n = run_condition(
        condition, query_prefix_tokens, max_new_tokens=DECODE_N, seed=seed
    )
    tpot = (e2e_n - ttft) / (DECODE_N - 1)
    return ttft, tpot


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(p * (len(values) - 1)))]


def summarize(name, values_s):
    values_ms = [v * 1000 for v in values_s]
    mean = statistics.mean(values_ms)
    print(
        f"  {name}: mean={mean:.2f}ms p50={percentile(values_ms, 0.50):.2f}ms "
        f"p75={percentile(values_ms, 0.75):.2f}ms p90={percentile(values_ms, 0.90):.2f}ms "
        f"p95={percentile(values_ms, 0.95):.2f}ms (n={len(values_ms)})"
    )
    return mean


def main():
    results = {}
    for total_context in TOTAL_CONTEXT_SIZES:
        query_prefix_tokens = total_context - DONOR_TOKENS
        print(
            f"\n=== total_context={total_context} "
            f"(query_prefix={query_prefix_tokens}, donor={DONOR_TOKENS}) ==="
        )
        results[total_context] = {}
        for condition in ("miss", "hit"):
            print(f"Warming up {condition} ({WARMUP_REPEATS} repeats, not counted)...")
            for i in range(WARMUP_REPEATS):
                one_trial(condition, query_prefix_tokens, seed=SEED + i)

            ttfts, tpots = [], []
            for i in range(MEASURED_REPEATS):
                ttft, tpot = one_trial(
                    condition, query_prefix_tokens, seed=SEED + 1000 + i
                )
                ttfts.append(ttft)
                tpots.append(tpot)

            print(f"{condition.upper()} (n={MEASURED_REPEATS}):")
            mean_ttft = summarize("TTFT", ttfts)
            mean_tpot = summarize("TPOT", tpots)
            results[total_context][condition] = {
                "ttft_ms": mean_ttft,
                "tpot_ms": mean_tpot,
            }

    print("\n=== Summary: TTFT speedup (miss/hit) and TPOT parity ===")
    for total_context, by_cond in results.items():
        miss_ttft = by_cond["miss"]["ttft_ms"]
        hit_ttft = by_cond["hit"]["ttft_ms"]
        miss_tpot = by_cond["miss"]["tpot_ms"]
        hit_tpot = by_cond["hit"]["tpot_ms"]
        print(
            f"total_context={total_context}: "
            f"TTFT speedup={miss_ttft / hit_ttft:.3f}x "
            f"(miss={miss_ttft:.2f}ms hit={hit_ttft:.2f}ms), "
            f"TPOT ratio={hit_tpot / miss_tpot:.3f}x "
            f"(miss={miss_tpot:.2f}ms hit={hit_tpot:.2f}ms)"
        )


if __name__ == "__main__":
    main()
