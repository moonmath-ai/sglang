"""Context-window sweep -- see `measure_context_sweep_mla.py` for the full rationale
and the single-chunk-cap caveat (unchanged here). This variant targets
`Moonlight-16B-A3B-Instruct`.

**Cannot reach the reported 8k threshold at all.** Moonlight's real
`max_position_embeddings` is 8192 (confirmed via its `config.json`, no YaRN/
`rope_scaling` to extend it) -- unlike DeepSeek-V2-Lite's YaRN-scaled 163840. The sweep
below is capped at 7800 total tokens (leaving headroom below 8192 for the TPOT
sub-call's extra `DECODE_N` generated tokens), so this script can only establish
"no benefit below 8k" on Moonlight, not confirm or refute what happens at or past it --
that question can only be answered on DeepSeek-V2-Lite in this fork's current fleet.

Not a CI test; run manually against an already-running fuzzy_match+ExactHash server:

    sglang serve --model-path moonshotai/Moonlight-16B-A3B-Instruct \\
      --radix-cache-backend fuzzy_match --fuzzy-match-provider ExactHash \\
      --host 127.0.0.1 --port 21000
    python test/manual/fuzzy_match/measure_context_sweep_mla_moonlight.py http://127.0.0.1:21000
"""

import random
import statistics
import sys

sys.path.insert(0, "/home/karthik/sglang-private/python")

from sglang.test.kl_test_utils import _flush_cache, _generate

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:21000"

SINK_TOKENS = 32
DONOR_TOKENS = 150
DECODE_N = 32
WARMUP_REPEATS = 3
MEASURED_REPEATS = 20
SEED = 20260802
# Moonlight-16B-A3B-Instruct: vocab_size=163840, bos_token_id=163584 -- HIGH leaves a
# 3500+-token buffer below the special-token boundary.
LOW, HIGH = 1000, 160000

# Capped well below Moonlight's real max_position_embeddings=8192 (no YaRN to extend
# it) -- see module docstring.
TOTAL_CONTEXT_SIZES = [200, 2000, 4000, 6000, 7800]


def rand_tokens(rng, n):
    return [rng.randint(LOW, HIGH) for _ in range(n)]


def run_condition(condition, query_prefix_tokens, max_new_tokens, seed):
    rng = random.Random(seed)
    donor_content = rand_tokens(rng, DONOR_TOKENS)
    query_prefix = rand_tokens(rng, query_prefix_tokens)
    if condition == "miss":
        tail = rand_tokens(rng, DONOR_TOKENS)
    else:
        tail = donor_content

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
