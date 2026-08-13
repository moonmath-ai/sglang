"""E2E correctness test: does ExactHashProvider's shifted-offset reuse
produce numerically consistent output vs. a fresh full recompute, on a
real MLA (Multi-Head Latent Attention) model?

Stage 1 (``test_exact_hash_shifted_offset_kl.py``) validated the mechanism
on Qwen2.5-7B, a full-RoPE model (``rotary_dim == head_dim``) with no
position-free slice at all — it proved the correction is *correct*, not
that it's *cheap in the way Irminsul's own MLA-native claim is*. This test
closes that gap: ``DeepSeek-V2-Lite-Chat`` is one of the Irminsul paper's
own four evaluated configurations, and its ``DeepseekV2AttentionMLA`` code
path is identical to GLM 5.2's — this is the template engineering extends
to GLM 5.2 on the real allocation (see ``POC_PLAN.md``'s Stage 2).

Same scenario-construction rationale as the MHA version: ``ExactHashProvider``
only extends a match through *consecutive* same-donor chunks (POC_PLAN.md
Stage 2's multi-chunk follow-up), so the registration prefix must be exactly
``SINK_TOKENS`` long and the query prefix must be *fully* exact-matched
beforehand -- otherwise the donor content isn't chunked "fresh from its own
start" on both the register and query sides, and real CDC's boundaries (which
depend on the preceding ``WINDOW_SIZE`` tokens of context, not just the donor
content itself) silently diverge between the two. See that file's docstring
for the full explanation, unchanged here.

**Must positively assert the correction path fired, not just check KL.**
Running this scenario against an MLA pool that ``FuzzyKVRealizer`` doesn't
support would silently no-op the realization (``pool_supported=False`` ->
``_free_realization_slots``, no correction attempted) rather than error —
meaning a KL-only check could pass for the wrong reason (plain
exact-prefix ``RadixCache`` behavior, never exercising the MLA correction
at all). Server stdout/stderr are captured and checked for the
``[FUZZY] Realized`` marker (only emitted once
``_realize_contiguous``/``_realize_segments`` actually executes a
correction) in addition to the KL comparison.
"""

import os
import random
import re
import tempfile
import unittest

from sglang.srt.mem_cache.fuzzy_match.chunker import MAX_CHUNK_TOKENS
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _get_input_logprobs_and_top1,
    compare_argmax_match_and_divergence,
    compare_kl_divergence,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"

SINK_TOKENS = 32  # must match chunker.SINK_TOKENS
NUM_SAMPLES = 8
DONOR_TOKENS = 400
QUERY_PREFIX_TOKENS = 200

# Comfortably bigger than chunker.MAX_CHUNK_TOKENS (512) -- no single
# content-defined chunk can ever span this whole donor, so recovering most
# of it requires ExactHashProvider's multi-chunk same-donor extension
# (POC_PLAN.md Stage 2's follow-up); the original Stage 1 provider only ever
# matched the unmatched tail's first chunk and was hard-capped at 512.
LARGE_DONOR_TOKENS = 700
LARGE_NUM_SAMPLES = 4
# DeepSeek-V2-Lite-Chat: vocab_size=102400, bos_token_id=100000,
# eos_token_id=100001 (confirmed via config.json) -- HIGH leaves a
# 1000-token buffer below the 100000 special-token boundary, mirroring the
# MHA test's margin above 0.
SYNTHETIC_TOKEN_LOW = 1000
SYNTHETIC_TOKEN_HIGH = 99000

# Starting point, not a value derived from the paper -- see Stage 2's plan
# for the decision rule on loosening this (rule out a real bug via the
# mscale unit test first, don't just loosen on a marginal failure).
ACC_THRESHOLDS = {MODEL: {"kl_div": 0.02}}

FUZZY_SUCCESS_MARKER = "[FUZZY RADIX] fuzzy match success"
FUZZY_REALIZED_MARKER = "[FUZZY] Realized"

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")


def _log_tail(log_paths, start_pos_by_path):
    """Concatenate everything written to any of ``log_paths`` since the
    corresponding recorded position — logger output location (stdout vs
    stderr) isn't asserted on, so both are scanned."""
    chunks = []
    for path in log_paths:
        with open(path) as f:
            f.seek(start_pos_by_path[path])
            chunks.append(f.read())
    return "\n".join(chunks)


def _log_positions(log_paths):
    return {path: os.path.getsize(path) for path in log_paths}


class TestExactHashShiftedOffsetMLA(CustomTestCase):
    STDOUT_PATH = os.path.join(
        tempfile.gettempdir(), "fuzzy_mla_shifted_offset_stdout.txt"
    )
    STDERR_PATH = os.path.join(
        tempfile.gettempdir(), "fuzzy_mla_shifted_offset_stderr.txt"
    )

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.stdout = open(cls.STDOUT_PATH, "w")
        cls.stderr = open(cls.STDERR_PATH, "w")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--radix-cache-backend",
                "fuzzy_match",
                "--fuzzy-match-provider",
                "ExactHash",
                # Resource envelope for shared-GPU boxes (template semantics
                # unaffected): bounded static pool, bounded concurrency, small
                # cuda-graph capture set.
                "--max-running-requests",
                "4",
                "--cuda-graph-max-bs",
                "8",
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "stdout"):
            cls.stdout.close()
        if hasattr(cls, "stderr"):
            cls.stderr.close()

    def test_shifted_offset_reuse_matches_full_recompute(self):
        # Synthetic, guaranteed-non-overlapping content — see module
        # docstring for why real dataset text doesn't work for this.
        rng = random.Random(20260730)

        def rand_tokens(n):
            return [
                rng.randint(SYNTHETIC_TOKEN_LOW, SYNTHETIC_TOKEN_HIGH) for _ in range(n)
            ]

        donor_contents = [rand_tokens(DONOR_TOKENS) for _ in range(NUM_SAMPLES)]
        query_prefixes = [rand_tokens(QUERY_PREFIX_TOKENS) for _ in range(NUM_SAMPLES)]

        _flush_cache(self.base_url)

        # Fixed SINK_TOKENS-length prefix: makes chunk_region_start land
        # exactly at donor content's start during registration (see module
        # docstring).
        sink_prefix = list(range(50000, 50000 + SINK_TOKENS))
        registration_prompts = [sink_prefix + d for d in donor_contents]
        _generate(self.base_url, registration_prompts, max_new_tokens=0)

        # Pre-cache each query prefix standalone, so the combined request
        # below gets a full *exact* match on it — leaving donor content as
        # the entire unmatched tail, chunked fresh from its own start.
        _generate(self.base_url, query_prefixes, max_new_tokens=0)

        log_paths = [self.STDOUT_PATH, self.STDERR_PATH]
        start = _log_positions(log_paths)

        query_prompts = [
            query_prefixes[i] + donor_contents[i] for i in range(NUM_SAMPLES)
        ]
        results = _generate(
            self.base_url,
            query_prompts,
            max_new_tokens=64,
            return_logprob=True,
        )
        self.assertEqual(len(results), NUM_SAMPLES)

        log_tail = _log_tail(log_paths, start)
        self.assertEqual(
            log_tail.count(FUZZY_SUCCESS_MARKER),
            NUM_SAMPLES,
            "expected every donor to be found via fuzzy match",
        )
        self.assertIn(
            FUZZY_REALIZED_MARKER,
            log_tail,
            "fuzzy match succeeded but the MLA correction path never fired — "
            "FuzzyKVRealizer.pool_supported is likely False for this pool "
            "(silently falling back to no-op realization), which would make "
            "the KL comparison below pass for the wrong reason (plain "
            "exact-prefix reuse, not the MLA delta-rotation correction)",
        )

        new_input_ids = []
        output_logprobs = []
        for i, result in enumerate(results):
            new_input_ids.append(query_prompts[i] + result["output_ids"])
            output_logprobs.append(_extract_output_logprobs(result))

        input_logprobs, reference_top1_ids = _get_input_logprobs_and_top1(
            self.base_url, new_input_ids, output_logprobs
        )

        compare_kl_divergence(
            input_logprobs,
            output_logprobs,
            ACC_THRESHOLDS,
            self.model,
            "test_shifted_offset_reuse_matches_full_recompute",
        )

        treatment_output_ids = [result["output_ids"] for result in results]
        compare_argmax_match_and_divergence(
            treatment_output_ids,
            reference_top1_ids,
            self.model,
            "test_shifted_offset_reuse_matches_full_recompute",
        )

    def test_shifted_offset_reuse_recovers_more_than_one_chunk(self):
        """(derived property) A donor span bigger than any single CDC chunk
        (``chunker.MAX_CHUNK_TOKENS`` = 512) must still be recovered beyond
        that cap: ``ExactHashProvider.match_on_prefix_miss`` walks every
        consecutive same-donor chunk, not just the unmatched tail's first
        one (POC_PLAN.md Stage 2's multi-chunk follow-up). Before that
        change, the realized span could never exceed one chunk's own size,
        so no donor content past position ~512 was ever reused regardless of
        how much more was available. Asserts on the ``cached=`` fuzzy-match
        log field directly (not just KL) so a future regression that
        silently reintroduces a single-chunk cap fails here even if KL still
        happens to pass on this content."""
        rng = random.Random(20260802)

        def rand_tokens(n):
            return [
                rng.randint(SYNTHETIC_TOKEN_LOW, SYNTHETIC_TOKEN_HIGH) for _ in range(n)
            ]

        donor_contents = [
            rand_tokens(LARGE_DONOR_TOKENS) for _ in range(LARGE_NUM_SAMPLES)
        ]
        query_prefixes = [
            rand_tokens(QUERY_PREFIX_TOKENS) for _ in range(LARGE_NUM_SAMPLES)
        ]

        _flush_cache(self.base_url)

        sink_prefix = list(range(50000, 50000 + SINK_TOKENS))
        registration_prompts = [sink_prefix + d for d in donor_contents]
        _generate(self.base_url, registration_prompts, max_new_tokens=0)
        _generate(self.base_url, query_prefixes, max_new_tokens=0)

        log_paths = [self.STDOUT_PATH, self.STDERR_PATH]
        start = _log_positions(log_paths)

        query_prompts = [
            query_prefixes[i] + donor_contents[i] for i in range(LARGE_NUM_SAMPLES)
        ]
        results = _generate(
            self.base_url,
            query_prompts,
            max_new_tokens=64,
            return_logprob=True,
        )
        self.assertEqual(len(results), LARGE_NUM_SAMPLES)

        log_tail = _log_tail(log_paths, start)
        self.assertEqual(
            log_tail.count(FUZZY_SUCCESS_MARKER),
            LARGE_NUM_SAMPLES,
            "expected every donor to be found via fuzzy match",
        )

        cached_counts = [int(m) for m in re.findall(r"cached=(\d+)", log_tail)]
        self.assertEqual(len(cached_counts), LARGE_NUM_SAMPLES)
        self.assertGreater(
            max(cached_counts),
            MAX_CHUNK_TOKENS,
            f"no sample recovered more than one chunk's worth "
            f"({MAX_CHUNK_TOKENS} tokens) -- multi-chunk extension isn't "
            f"firing: cached_counts={cached_counts}",
        )

        new_input_ids = []
        output_logprobs = []
        for i, result in enumerate(results):
            new_input_ids.append(query_prompts[i] + result["output_ids"])
            output_logprobs.append(_extract_output_logprobs(result))

        input_logprobs, reference_top1_ids = _get_input_logprobs_and_top1(
            self.base_url, new_input_ids, output_logprobs
        )

        compare_kl_divergence(
            input_logprobs,
            output_logprobs,
            ACC_THRESHOLDS,
            self.model,
            "test_shifted_offset_reuse_recovers_more_than_one_chunk",
        )

        treatment_output_ids = [result["output_ids"] for result in results]
        compare_argmax_match_and_divergence(
            treatment_output_ids,
            reference_top1_ids,
            self.model,
            "test_shifted_offset_reuse_recovers_more_than_one_chunk",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
