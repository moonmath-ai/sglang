"""Tests whether the delta-rotation correction still holds under a quantized
KV cache (PIC_IRMINSUL.md's "Open questions/risks" -- unverified, flagged as
important since production deployments may want FP8 KV cache for capacity).
Not a CI test (the result, at time of writing, is a confirmed failure, not a
passing gate) -- run manually against an already-running server:

    python test/manual/fuzzy_match/measure_fp8_kv_cache_correctness.py <base_url>

Launch the server with:
    sglang serve --model-path Qwen/Qwen2.5-7B-Instruct-AWQ \\
      --radix-cache-backend fuzzy_match --fuzzy-match-provider ExactHash \\
      --kv-cache-dtype fp8_e5m2 --host 127.0.0.1 --port 21030

Runs two teacher-forced KL-divergence checks against the SAME fp8-KV-cache
server:
  1. CONTROL: plain fresh prompts, no donor/fuzzy match involved at all --
     isolates whether fp8_e5m2 KV cache is noisy *on its own* (normal
     forward-pass RoPE happens pre-quantization; the KV buffer is only
     quantized once, at write time).
  2. WITH FUZZY MATCH: identical construction to
     test_exact_hash_shifted_offset_kl.py -- exercises the delta-rotation
     correction, which necessarily operates on an *already-quantized* cached
     K vector (that's inherent to correcting cached KV rather than
     recomputing it).

Root cause, confirmed by reading the code (not just observing the number):
layers/rotary_embedding/utils.py's apply_rotary_emb/reverse_rotary_emb both
do `cos = cos.unsqueeze(-2).to(x.dtype)` before the rotation multiply -- when
`x` is a torch.float8_e5m2 K vector (already-quantized, read straight from
the pool), the entire rotation (both the "undo old position" and "apply new
position" steps) happens in native 8-bit float arithmetic. e5m2 has only 2
mantissa bits, so cos/sin values get rounded to an extremely coarse grid
before ever being multiplied. Normal (non-fuzzy) forward-pass RoPE never
hits this: Q/K are rotated in full compute precision *before* being written
into the (quantized) KV pool, so quantization only ever touches the
post-rotation result once. This POC's correction is the first thing to do
floating-point rotation math on an *already-quantized* value, and it isn't
upcasting first.
"""

import random
import sys

sys.path.insert(0, "/home/karthik/sglang-private/python")

from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _get_input_logprobs,
    compare_kl_divergence,
)

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:21030"
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
SINK_TOKENS = 32
NUM_SAMPLES = 8
DONOR_TOKENS = 400
QUERY_PREFIX_TOKENS = 200
LOW, HIGH = 1000, 140000
ACC_THRESHOLDS = {MODEL: {"kl_div": 0.02}}


def rand_tokens(rng, n):
    return [rng.randint(LOW, HIGH) for _ in range(n)]


def _kl_for(new_input_ids, output_logprobs, label):
    input_logprobs = _get_input_logprobs(BASE_URL, new_input_ids, output_logprobs)
    try:
        compare_kl_divergence(
            input_logprobs, output_logprobs, ACC_THRESHOLDS, MODEL, label
        )
        print(f"{label}: PASSED")
    except AssertionError as e:
        print(f"{label}: FAILED -- {e}")


def run_control(rng):
    """No donor, no shifted offset, no fuzzy match -- isolates baseline fp8
    KV cache noise, independent of this POC's correction code."""
    prompts = [rand_tokens(rng, QUERY_PREFIX_TOKENS + DONOR_TOKENS) for _ in range(NUM_SAMPLES)]
    _flush_cache(BASE_URL)
    results = _generate(BASE_URL, prompts, max_new_tokens=64, return_logprob=True)
    new_input_ids = [prompts[i] + r["output_ids"] for i, r in enumerate(results)]
    output_logprobs = [_extract_output_logprobs(r) for r in results]
    _kl_for(new_input_ids, output_logprobs, "fp8_kv_cache_control_no_fuzzy")


def run_with_fuzzy_match(rng):
    """Identical construction to test_exact_hash_shifted_offset_kl.py --
    exercises the delta-rotation correction on an already-quantized K."""
    donor_contents = [rand_tokens(rng, DONOR_TOKENS) for _ in range(NUM_SAMPLES)]
    query_prefixes = [rand_tokens(rng, QUERY_PREFIX_TOKENS) for _ in range(NUM_SAMPLES)]

    _flush_cache(BASE_URL)
    sink_prefix = list(range(50000, 50000 + SINK_TOKENS))
    _generate(BASE_URL, [sink_prefix + d for d in donor_contents], max_new_tokens=0)
    _generate(BASE_URL, query_prefixes, max_new_tokens=0)

    query_prompts = [query_prefixes[i] + donor_contents[i] for i in range(NUM_SAMPLES)]
    results = _generate(BASE_URL, query_prompts, max_new_tokens=64, return_logprob=True)
    new_input_ids = [query_prompts[i] + r["output_ids"] for i, r in enumerate(results)]
    output_logprobs = [_extract_output_logprobs(r) for r in results]
    _kl_for(new_input_ids, output_logprobs, "fp8_kv_cache_with_fuzzy_match")


def main():
    rng = random.Random(20260726)
    run_control(rng)
    run_with_fuzzy_match(rng)


if __name__ == "__main__":
    main()
