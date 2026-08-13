"""E2E correctness test: same as ``test_exact_hash_shifted_offset_kl_mla.py``
but on ``Moonlight-16B-A3B-Instruct`` — a second, independent MLA target.

Moonlight is ``DeepseekV3ForCausalLM``, a literal subclass of
``DeepseekV2ForCausalLM`` in the same model file (same ``attn_mqa``, no new
code paths exercised) — but it's architecturally distinct in two relevant
ways: (1) it's one of the Irminsul paper's own three headline evaluated
models (``moonshotai/Moonlight-16B-A3B-Instruct``, per the paper's own
citation), independent of DeepSeek-V2-Lite; (2) its config has **no
``rope_scaling`` at all** (plain ``rope_theta=50000``, no YaRN) — confirms
the mechanism doesn't depend on any YaRN-specific behavior the first MLA
test happened to exercise (and trivially sidesteps the `mscale` question
entirely, since there's no YaRN scaling to have an `mscale` in the first
place).

See ``test_exact_hash_shifted_offset_kl_mla.py`` for the full scenario-
construction rationale (unchanged here) and why the correction path firing
must be positively asserted, not just inferred from KL passing.
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

MODEL = "moonshotai/Moonlight-16B-A3B-Instruct"

SINK_TOKENS = 32  # must match chunker.SINK_TOKENS
NUM_SAMPLES = 8
DONOR_TOKENS = 400
QUERY_PREFIX_TOKENS = 200

# Comfortably bigger than chunker.MAX_CHUNK_TOKENS (512) -- see the
# DeepSeek-V2-Lite variant's comment for why. Moonlight's real
# max_position_embeddings=8192 comfortably fits SINK_TOKENS + this +
# QUERY_PREFIX_TOKENS.
LARGE_DONOR_TOKENS = 700
LARGE_NUM_SAMPLES = 4
# Moonlight-16B-A3B-Instruct: vocab_size=163840, bos_token_id=163584,
# eos_token_id=163586 (confirmed via config.json) -- HIGH leaves a
# 3500+-token buffer below the 163584 special-token boundary.
SYNTHETIC_TOKEN_LOW = 1000
SYNTHETIC_TOKEN_HIGH = 160000

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


class TestExactHashShiftedOffsetMLAMoonlight(CustomTestCase):
    STDOUT_PATH = os.path.join(
        tempfile.gettempdir(), "fuzzy_mla_moonlight_shifted_offset_stdout.txt"
    )
    STDERR_PATH = os.path.join(
        tempfile.gettempdir(), "fuzzy_mla_moonlight_shifted_offset_stderr.txt"
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
        # Synthetic, guaranteed-non-overlapping content — see the
        # DeepSeek-V2-Lite variant's docstring for why real dataset text
        # doesn't work for this.
        rng = random.Random(20260730)

        def rand_tokens(n):
            return [
                rng.randint(SYNTHETIC_TOKEN_LOW, SYNTHETIC_TOKEN_HIGH) for _ in range(n)
            ]

        donor_contents = [rand_tokens(DONOR_TOKENS) for _ in range(NUM_SAMPLES)]
        query_prefixes = [rand_tokens(QUERY_PREFIX_TOKENS) for _ in range(NUM_SAMPLES)]

        _flush_cache(self.base_url)

        # Fixed SINK_TOKENS-length prefix: makes chunk_region_start land
        # exactly at donor content's start during registration.
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
        """(derived property) See the DeepSeek-V2-Lite variant's docstring
        for the full rationale -- a donor span bigger than any single CDC
        chunk (``chunker.MAX_CHUNK_TOKENS`` = 512) must still be recovered
        beyond that cap via ``ExactHashProvider.match_on_prefix_miss``'s
        multi-chunk same-donor extension. Asserts on the ``cached=``
        fuzzy-match log field directly (not just KL) so a future regression
        that silently reintroduces a single-chunk cap fails here even if KL
        still happens to pass on this content."""
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
