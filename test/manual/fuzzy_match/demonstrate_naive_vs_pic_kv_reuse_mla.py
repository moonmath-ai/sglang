"""Compares the Irminsul-paper "naive reuse" baseline (donor KV copied
verbatim, zero positional correction) against PIC's RoPE delta-rotation
correction, at the same shifted-offset scenario used by
``test_exact_hash_shifted_offset_kl_mla.py``. Not a CI test -- the K-vector-
level guarantee that correction matters is already pinned down
deterministically by
``test_naive_reuse_produces_large_rel_l2_error`` in
``test/registered/unit/mem_cache/fuzzy_match/test_mla_rope_correction.py``.
This script instead reports the end-to-end effect, which turns out to be
workload- and shift-dependent for MLA models: probed across three shift
magnitudes on both DeepSeek-V2-Lite-Chat and Moonlight-16B-A3B-Instruct, the
naive-vs-corrected avg-KL gap ranged from ~1.05x (statistically
indistinguishable) to ~5x, non-monotonically in shift size -- consistent
with the paper's own finding that naive reuse is within noise of PIC on at
least one DeepSeek-V2-Lite workload (GovReport, ±0.02 KL). MLA's compressed
latent (``k_nope``, position-free) carries most of the attention-score
magnitude, so corrupting only the ``k_rope`` slice doesn't always surface
strongly in the final output distribution -- this is presumptive, not
independently confirmed here.

Run against two separately-launched servers (naive reuse requires setting
``SGLANG_TEST_FUZZY_NAIVE_KV_REUSE=true`` *before* the server process
starts, so it can't be toggled against one already-running server):

    # Corrected (PIC) path
    sglang serve --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \\
      --radix-cache-backend fuzzy_match --fuzzy-match-provider ExactHash \\
      --host 127.0.0.1 --port 21000
    python test/manual/fuzzy_match/demonstrate_naive_vs_pic_kv_reuse_mla.py \\
      http://127.0.0.1:21000 deepseek-ai/DeepSeek-V2-Lite-Chat

    # Naive-reuse baseline
    SGLANG_TEST_FUZZY_NAIVE_KV_REUSE=true sglang serve \\
      --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \\
      --radix-cache-backend fuzzy_match --fuzzy-match-provider ExactHash \\
      --host 127.0.0.1 --port 21001
    python test/manual/fuzzy_match/demonstrate_naive_vs_pic_kv_reuse_mla.py \\
      http://127.0.0.1:21001 deepseek-ai/DeepSeek-V2-Lite-Chat
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))

from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _get_input_logprobs,
    compute_avg_kl,
)

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:21000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek-ai/DeepSeek-V2-Lite-Chat"

SINK_TOKENS = 32
NUM_SAMPLES = 8
DONOR_TOKENS = 400
# Vocab margins per model, matching the corresponding E2E test file.
VOCAB_MARGINS = {
    "deepseek-ai/DeepSeek-V2-Lite-Chat": (1000, 99000),
    "moonshotai/Moonlight-16B-A3B-Instruct": (1000, 160000),
}
LOW, HIGH = VOCAB_MARGINS.get(MODEL, (1000, 99000))


def run_scenario(query_prefix_tokens):
    rng = random.Random(20260730)

    def rand_tokens(n):
        return [rng.randint(LOW, HIGH) for _ in range(n)]

    donor_contents = [rand_tokens(DONOR_TOKENS) for _ in range(NUM_SAMPLES)]
    query_prefixes = [rand_tokens(query_prefix_tokens) for _ in range(NUM_SAMPLES)]

    _flush_cache(BASE_URL)

    sink_prefix = list(range(50000, 50000 + SINK_TOKENS))
    registration_prompts = [sink_prefix + d for d in donor_contents]
    _generate(BASE_URL, registration_prompts, max_new_tokens=0)
    _generate(BASE_URL, query_prefixes, max_new_tokens=0)

    query_prompts = [query_prefixes[i] + donor_contents[i] for i in range(NUM_SAMPLES)]
    results = _generate(BASE_URL, query_prompts, max_new_tokens=64, return_logprob=True)

    new_input_ids = []
    output_logprobs = []
    for i, result in enumerate(results):
        new_input_ids.append(query_prompts[i] + result["output_ids"])
        output_logprobs.append(_extract_output_logprobs(result))

    input_logprobs = _get_input_logprobs(BASE_URL, new_input_ids, output_logprobs)
    _, avg_kl = compute_avg_kl(input_logprobs, output_logprobs)
    return avg_kl


if __name__ == "__main__":
    for shift in (168, 968, 2968):
        avg_kl = run_scenario(shift + SINK_TOKENS)
        print(f"[{MODEL}] shift={shift} avg_kl={avg_kl:.6f}")
