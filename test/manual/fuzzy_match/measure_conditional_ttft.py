"""Conditional (per-hit) TTFT measurement (PIC_IRMINSUL.md Section 3
Measurements) — isolates the mechanism's own per-hit cost from whatever the
corpus recovery rate happens to be. Not a CI test; run manually against an
already-running server:

    python test/manual/fuzzy_match/measure_conditional_ttft.py <base_url>

For each repeat: HIT case sends query_prefix + a *registered* donor span
(gets an ExactHash correction); MISS case sends query_prefix + a
same-length, never-registered span (guaranteed miss). Same total prefill
size in both cases — isolates "was this span skipped or not", not corpus
recovery rate (that's measure_recovery_rate.py's job).

TTFT itself isn't in the response meta_info, so this measures wall-clock
time for a max_new_tokens=1 request — for a single generated token, that's
dominated by prefill, a reasonable proxy without needing a new metrics field.
"""

import random
import statistics
import sys
import time

sys.path.insert(0, "/home/karthik/sglang-private/python")

from sglang.test.kl_test_utils import _flush_cache, _generate

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:21000"
SINK_TOKENS = 32
# 150 tokens lands as a single CDC chunk most of the time (~63% empirically)
# — Stage 1's "check chunks[0] only" limitation means anything beyond the
# first matched chunk falls through to a genuine miss, so a larger
# DONOR_TOKENS (e.g. 400) mostly measures a partial hit diluted by a real
# miss on the remainder, not the mechanism's cleanest case.
DONOR_TOKENS = 150
QUERY_PREFIX_TOKENS = 200
LOW, HIGH = 1000, 140000
WARMUP_REPEATS = 3
MEASURED_REPEATS = 10
SEED = 20260726


def rand_tokens(rng, n):
    return [rng.randint(LOW, HIGH) for _ in range(n)]


def time_generate(base_url, prompt):
    start = time.perf_counter()
    _generate(base_url, [prompt], max_new_tokens=1)
    return time.perf_counter() - start


def one_trial(rng):
    donor_content = rand_tokens(rng, DONOR_TOKENS)
    query_prefix = rand_tokens(rng, QUERY_PREFIX_TOKENS)
    miss_content = rand_tokens(rng, DONOR_TOKENS)  # same length, never registered

    _flush_cache(BASE_URL)
    sink_prefix = list(range(50000, 50000 + SINK_TOKENS))
    _generate(BASE_URL, [sink_prefix + donor_content], max_new_tokens=1)
    _generate(BASE_URL, [query_prefix], max_new_tokens=1)

    hit_time = time_generate(BASE_URL, query_prefix + donor_content)
    miss_time = time_generate(BASE_URL, query_prefix + miss_content)
    return hit_time, miss_time


def summarize(name, values):
    values = sorted(values)
    n = len(values)
    mean = statistics.mean(values)
    p50 = values[n // 2]
    p95 = values[min(n - 1, int(0.95 * (n - 1)))]
    print(f"{name}: mean={mean*1000:.1f}ms p50={p50*1000:.1f}ms p95={p95*1000:.1f}ms "
          f"min={min(values)*1000:.1f}ms max={max(values)*1000:.1f}ms (n={n})")
    return mean


def main():
    rng = random.Random(SEED)

    print(f"Warming up ({WARMUP_REPEATS} repeats, not counted)...")
    for _ in range(WARMUP_REPEATS):
        one_trial(rng)

    print(f"Measuring ({MEASURED_REPEATS} repeats)...")
    hit_times, miss_times = [], []
    for i in range(MEASURED_REPEATS):
        hit_t, miss_t = one_trial(rng)
        hit_times.append(hit_t)
        miss_times.append(miss_t)
        print(f"  repeat {i}: hit={hit_t*1000:.1f}ms miss={miss_t*1000:.1f}ms")

    print()
    mean_hit = summarize("HIT  (conditional, PIC-corrected)", hit_times)
    mean_miss = summarize("MISS (baseline, full recompute)  ", miss_times)
    print(f"\nSpeedup (miss/hit): {mean_miss / mean_hit:.2f}x")


if __name__ == "__main__":
    main()
