"""Unit tests for fuzzy-match provider integration edges.

Covers the provider factory contract, config validation at construction,
and donor lock_ref accounting across multiple concurrent recipients.

ExactHash-only (Irminsul PIC) port: the SemBlend adapter seams and result
translation tests from the original PR went away with the
SemanticEmbedding provider.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import threading
import unittest
from array import array
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.exact_hash_provider import ExactHashProvider
from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import (
    FuzzyMatchProvider,
    FuzzyMatchResult,
    create_fuzzy_match_provider,
)
from sglang.srt.mem_cache.fuzzy_match.fuzzy_radix_cache import FuzzyRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.test_utils import CustomTestCase


class TestCreateFuzzyMatchProvider(CustomTestCase):
    def test_returns_none_when_disabled(self):
        """A config with fuzzy matching disabled must not build a provider;
        a regression here silently turns fuzzy matching always-on."""
        self.assertIsNone(create_fuzzy_match_provider(FuzzyMatchConfig()))

    def test_builds_exact_hash_when_enabled(self):
        """The factory must route the default provider name to
        ExactHashProvider; a broken lookup leaves the backend exact-only
        despite fuzzy being enabled."""
        provider = create_fuzzy_match_provider(
            FuzzyMatchConfig(enable_fuzzy_match=True)
        )
        self.assertIsInstance(provider, ExactHashProvider)


class TestFuzzyMatchConfig(CustomTestCase):
    def test_validation_runs_at_construction(self):
        """Out-of-range values must be rejected when the struct is built.

        Guards the msgspec.Struct migration: if ``__post_init__`` stops being
        invoked on construction, invalid CLI values pass through silently and
        only fail deep inside the provider at runtime.
        """
        with self.assertRaises(ValueError):
            FuzzyMatchConfig(fuzzy_min_match_length=0)
        with self.assertRaises(ValueError):
            FuzzyMatchConfig(fuzzy_match_provider="NotAProvider")
        with self.assertRaises(ValueError):
            FuzzyMatchConfig(fuzzy_match_provider="SemanticEmbedding")
        with self.assertRaises(ValueError):
            FuzzyMatchConfig(fuzzy_min_suffix_tokens=-1)


class _DonorScriptedProvider(FuzzyMatchProvider):
    """Provider that answers every miss with one pre-built result."""

    def __init__(self, config, result):
        super().__init__(config)
        self._result = result

    def cache_on_request_finished(
        self,
        request,
        token_ids,
        kv_cache,
        cache_start_pos,
        cache_end_pos,
        radix_tree=None,
    ):
        return False

    def match_on_prefix_miss(
        self, prompt_token_ids, already_matched_len, request=None, extra_key=None
    ):
        return self._result


class _DeviceOnlyAllocator:
    device = "cpu"

    def __init__(self):
        self._next = 1000
        self._lock = threading.Lock()

    def alloc(self, size: int):
        with self._lock:
            start = self._next
            self._next += size
        return torch.arange(start, start + size, dtype=torch.int64)

    def free(self, indices):
        return None


class TestFuzzyDonorLockAccounting(CustomTestCase):
    def test_concurrent_matches_lock_same_donor_node(self):
        """N recipients matching the same donor must each take exactly one
        lock_ref pin (net N, and net zero after release). A regression that
        skips the pin for already-locked donors -- or double-releases -- lets
        LRU eviction free donor KV mid-forward."""
        cache = FuzzyRadixCache(
            params=CacheInitParams(
                disable=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=_DeviceOnlyAllocator(),
                page_size=1,
            )
        )
        donor_insert = cache.insert(
            InsertParams(
                key=RadixKey(token_ids=array("q", [1, 2, 3, 4]), extra_key=None),
                value=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            )
        )
        donor = donor_insert.last_device_node

        config = FuzzyMatchConfig(
            enable_fuzzy_match=True,
            fuzzy_min_match_length=1,
            cache_fuzzy_results=False,
            fuzzy_min_suffix_tokens=0,
        )
        scripted = FuzzyMatchResult(
            cached_token_count=2,
            cached_token_ids=[1, 2],
            prompt_token_count=3,
            kv_cache_indices=torch.tensor([10, 11], dtype=torch.int64),
            position_offset=0,
            # Non-aligned (donor positions differ from the zero-length exact
            # anchor) so the match takes the realization path rather than
            # being dropped as position-aligned.
            cached_start_pos=2,
            donor_last_node_id=donor.id,
        )
        cache.init_fuzzy_match(
            config=config,
            provider=_DonorScriptedProvider(config=config, result=scripted),
        )

        reqs = [
            SimpleNamespace(
                rid=f"req-{i}",
                fuzzy_donor_node=None,
                fuzzy_realized_locs=None,
                fuzzy_match_result=None,
                fuzzy_donor_align_pos=None,
            )
            for i in range(8)
        ]

        def run_match(req):
            result = cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(token_ids=array("q", [90, 91, 92]), extra_key=None),
                    req=req,
                )
            )
            self.assertEqual(result.fuzzy_matched_len, 2)
            self.assertIs(req.fuzzy_donor_node, donor)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(run_match, reqs))

        self.assertEqual(donor.lock_ref, len(reqs))

        for req in reqs:
            cache.dec_lock_ref(req.fuzzy_donor_node)
            req.fuzzy_donor_node = None

        self.assertEqual(donor.lock_ref, 0)


if __name__ == "__main__":
    unittest.main()
