"""GPU energy measurement (Irminsul paper's Table 1) -- see
``measure_energy_baseline_mla.py`` for the full rationale and the batched-
energy-read methodology (unchanged here; both live in this file too). This
variant targets ``Moonlight-16B-A3B-Instruct``, the second, independent MLA
target this fork validated (no YaRN, plain rope_theta=50000).

Not a CI test; run manually against an already-running *plain* server (no
fuzzy_match backend):

    sglang serve --model-path moonshotai/Moonlight-16B-A3B-Instruct \\
      --host 127.0.0.1 --port 21000
    python test/manual/fuzzy_match/measure_energy_baseline_mla_moonlight.py http://127.0.0.1:21000
"""

import random
import sys

sys.path.insert(0, "/home/karthik/sglang-private/python")

import pynvml

from sglang.test.kl_test_utils import _flush_cache, _generate

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:21000"

SEQ_LEN = 4096
HALF = SEQ_LEN // 2
# Moonlight-16B-A3B-Instruct: vocab_size=163840, bos_token_id=163584 --
# HIGH leaves a 3500+-token buffer below the special-token boundary.
LOW, HIGH = 1000, 160000
WARMUP_REPEATS = 3
MEASURED_REPEATS = 10
SEED = 20260730


def rand_tokens(rng, n):
    return [rng.randint(LOW, HIGH) for _ in range(n)]


def energy_of_calls(handle, calls):
    before_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    for prompt in calls:
        _generate(BASE_URL, [prompt], max_new_tokens=1)
    after_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    return (after_mj - before_mj) / len(calls)


def measure_miss(handle, rng, repeats):
    prompts = [rand_tokens(rng, SEQ_LEN) for _ in range(repeats)]
    _flush_cache(BASE_URL)
    return energy_of_calls(handle, prompts)


def measure_exact_hit(handle, rng, repeats):
    hit_prompt = rand_tokens(rng, SEQ_LEN)
    _flush_cache(BASE_URL)
    _generate(BASE_URL, [hit_prompt], max_new_tokens=1)  # register
    return energy_of_calls(handle, [hit_prompt] * repeats)


def measure_partial(handle, rng, repeats):
    partial_prefix = rand_tokens(rng, HALF)
    _flush_cache(BASE_URL)
    _generate(BASE_URL, [partial_prefix], max_new_tokens=1)  # register half prefix
    prompts = [partial_prefix + rand_tokens(rng, HALF) for _ in range(repeats)]
    return energy_of_calls(handle, prompts)


def main():
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        rng = random.Random(SEED)

        print(f"Warming up ({WARMUP_REPEATS} repeats, not counted)...")
        measure_miss(handle, rng, WARMUP_REPEATS)
        measure_exact_hit(handle, rng, WARMUP_REPEATS)
        measure_partial(handle, rng, WARMUP_REPEATS)

        print(f"Measuring ({MEASURED_REPEATS} repeats, batched per cell)...")
        mean_miss = measure_miss(handle, rng, MEASURED_REPEATS)
        mean_hit = measure_exact_hit(handle, rng, MEASURED_REPEATS)
        mean_partial = measure_partial(handle, rng, MEASURED_REPEATS)

        print(f"MISS       (full recompute)    : mean={mean_miss:.1f}mJ")
        print(f"EXACT_HIT  (full prefix reuse) : mean={mean_hit:.1f}mJ")
        print(f"PARTIAL    (~50% prefix reuse) : mean={mean_partial:.1f}mJ")
        print(
            f"\nSavings (exact hit vs miss):   {(1 - mean_hit / mean_miss) * 100:.1f}%"
        )
        print(
            f"Savings (partial hit vs miss): {(1 - mean_partial / mean_miss) * 100:.1f}%"
        )
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
