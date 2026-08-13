"""Energy analogue of `measure_context_sweep_mla.py` -- same shifted-offset PIC-hit
scenario (donor registered at one position, reused at a different one via
fuzzy_match+ExactHash), same context-size sweep, same single-chunk-cap caveat (see
that file's docstring for the full rationale -- unchanged here), but measuring GPU
energy per request instead of wall-clock latency.

Unlike `measure_energy_baseline_mla.py` (Part 4's Table-1-style baseline, which
deliberately used *plain* RadixCache and a same-offset hit -- "what's a normal cache
hit worth," zero fuzzy_match code involved), this script measures what PIC's own
per-hit mechanism costs in energy terms, mirroring `measure_conditional_ttft_mla.py`'s
question but for joules instead of milliseconds.

Energy is read via `pynvml.nvmlDeviceGetTotalEnergyConsumption`, batched across
`MEASURED_REPEATS` calls per condition per size (the counter's hardware sampling only
ticks every ~100ms -- see `measure_energy_baseline_mla.py`'s docstring for the full
explanation of why per-call reads alias badly). Each repeat inside a HIT batch uses a
*freshly drawn* donor (own sink-anchored registration) rather than reusing the same
donor byte-for-byte across repeats -- reusing identical content would let the second
repeat's full prompt become a plain *exact*-prefix hit after the first repeat's
request-finish auto-registers it, silently switching what's being measured partway
through the batch.

Not a CI test; run manually against an already-running fuzzy_match+ExactHash server:

    sglang serve --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \\
      --radix-cache-backend fuzzy_match --fuzzy-match-provider ExactHash \\
      --host 127.0.0.1 --port 21000
    python test/manual/fuzzy_match/measure_energy_context_sweep_mla.py http://127.0.0.1:21000
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))

import pynvml

from sglang.test.kl_test_utils import _flush_cache, _generate

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:21000"

SINK_TOKENS = 32
DONOR_TOKENS = 150
WARMUP_REPEATS = 3
MEASURED_REPEATS = 10
SEED = 20260802
LOW, HIGH = 1000, 99000  # DeepSeek-V2-Lite-Chat vocab_size=102400

TOTAL_CONTEXT_SIZES = [200, 2000, 8000, 16000, 32000]


def rand_tokens(rng, n):
    return [rng.randint(LOW, HIGH) for _ in range(n)]


def energy_of_calls(handle, prompts):
    before_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    for prompt in prompts:
        _generate(BASE_URL, [prompt], max_new_tokens=1)
    after_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    return (after_mj - before_mj) / len(prompts)


def measure_condition(handle, condition, total_context, repeats):
    query_prefix_tokens = total_context - DONOR_TOKENS
    rng = random.Random(SEED + total_context + hash(condition) % 1000)
    query_prefix = rand_tokens(rng, query_prefix_tokens)

    _flush_cache(BASE_URL)
    _generate(BASE_URL, [query_prefix], max_new_tokens=0)  # pre-cache the shared prefix

    if condition == "miss":
        prompts = [
            query_prefix + rand_tokens(rng, DONOR_TOKENS) for _ in range(repeats)
        ]
    else:
        sink_prefix = list(range(50000, 50000 + SINK_TOKENS))
        donors = [rand_tokens(rng, DONOR_TOKENS) for _ in range(repeats)]
        for d in donors:
            _generate(BASE_URL, [sink_prefix + d], max_new_tokens=0)
        prompts = [query_prefix + d for d in donors]

    return energy_of_calls(handle, prompts)


def main():
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        results = {}
        for total_context in TOTAL_CONTEXT_SIZES:
            print(f"\n=== total_context={total_context} ===")
            print(f"Warming up ({WARMUP_REPEATS} repeats, not counted)...")
            measure_condition(handle, "miss", total_context, WARMUP_REPEATS)
            measure_condition(handle, "hit", total_context, WARMUP_REPEATS)

            print(f"Measuring ({MEASURED_REPEATS} repeats, batched per condition)...")
            mean_miss = measure_condition(
                handle, "miss", total_context, MEASURED_REPEATS
            )
            mean_hit = measure_condition(handle, "hit", total_context, MEASURED_REPEATS)
            print(f"MISS: mean={mean_miss:.2f}mJ  HIT: mean={mean_hit:.2f}mJ")
            results[total_context] = (mean_miss, mean_hit)

        print("\n=== Summary: energy savings (miss vs hit) ===")
        for total_context, (mean_miss, mean_hit) in results.items():
            savings_pct = (1 - mean_hit / mean_miss) * 100
            print(
                f"total_context={total_context}: miss={mean_miss:.2f}mJ "
                f"hit={mean_hit:.2f}mJ savings={savings_pct:.1f}%"
            )
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
