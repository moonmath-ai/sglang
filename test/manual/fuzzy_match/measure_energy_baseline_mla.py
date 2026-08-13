"""GPU energy measurement (Irminsul paper's Table 1) -- what a cache hit is
actually worth in joules, independent of any fuzzy_match/PIC code. This
measures *plain* SGLang behavior (no ``--radix-cache-backend``/
``--fuzzy-match-provider`` flags at all, i.e. the default ``RadixCache``):
what's financially at stake from an exact-prefix cache hit, not Irminsul's
own per-hit correction cost (that's ``measure_conditional_ttft_mla.py``'s
job). The paper's own Table 1 pairs DeepSeek-V2-Lite with JoyAI-Flash, which
this fork has no access to -- this reuses the fork's own two already-
validated MLA targets (this file: DeepSeek-V2-Lite-Chat) instead of matching
the paper's exact pairing.

Not a CI test; run manually against an already-running *plain* server (no
fuzzy_match backend):

    sglang serve --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \\
      --host 127.0.0.1 --port 21000
    python test/manual/fuzzy_match/measure_energy_baseline_mla.py http://127.0.0.1:21000

Three cells at seq_len=4096: MISS (fresh never-seen prompt, full recompute),
EXACT_HIT (same prompt pre-registered, reissued byte-identical), PARTIAL
(2048 pre-cached + 2048 fresh tokens, ~50% hit via plain RadixCache prefix
matching -- no fuzzy match needed since the shared span is at the same
offset in both requests). 3 warmup + 10 timed repeats per cell.

Energy is sampled via ``pynvml.nvmlDeviceGetTotalEnergyConsumption`` (a
monotonic hardware mJ counter), confirmed to work with no permission issues
for a non-root user on this box. Confirmed empirically that this counter's
internal hardware sampling only ticks roughly every ~100ms (measured via
repeated back-to-back reads) -- individual EXACT_HIT/PARTIAL requests are
often *faster* than that (near-pure decode, minimal/no real prefill), so a
single before/after read wrapping just one such request frequently reads a
stale 0 and defers the real energy to whatever read happens next. To avoid
that bias, each cell's ``MEASURED_REPEATS`` requests are wrapped in a single
before/after energy read spanning the whole batch (cumulative duration much
larger than one tick), then divided by the repeat count -- exact by
telescoping (sum of true per-request energies always equals the batch's
total delta), unlike per-request reads.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))

import pynvml

from sglang.test.kl_test_utils import _flush_cache, _generate

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:21000"

SEQ_LEN = 4096
HALF = SEQ_LEN // 2
LOW, HIGH = 1000, 99000  # DeepSeek-V2-Lite-Chat vocab_size=102400
WARMUP_REPEATS = 3
MEASURED_REPEATS = 10
SEED = 20260730


def rand_tokens(rng, n):
    return [rng.randint(LOW, HIGH) for _ in range(n)]


def energy_of_calls(handle, calls):
    """Wrap a batch of ``_generate`` calls in one energy read -- see module
    docstring for why batching (not per-call reads) is needed here."""
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
    # Fresh suffix per repeat -- reusing the same suffix would let the
    # second repeat's identical prompt turn into a full exact hit instead
    # of staying at ~50%.
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
