"""Unit tests for srt/mem_cache/fuzzy_match/exact_hash_provider.py.

No server, no model loading — pure provider logic against a fake Req.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.fuzzy_match.chunker import SINK_TOKENS, Chunk
from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.exact_hash_provider import (
    ExactHashProvider,
    _ChunkEntry,
)
from sglang.test.test_utils import CustomTestCase


def _fake_req(rid: str, extra_key=None):
    return types.SimpleNamespace(rid=rid, extra_key=extra_key)


def _provider() -> ExactHashProvider:
    return ExactHashProvider(
        FuzzyMatchConfig(enable_fuzzy_match=True, fuzzy_match_provider="ExactHash")
    )


class TestExactHashProviderMatching(CustomTestCase):
    def test_exact_match_at_shifted_offset(self):
        """The entire point of the mechanism: content registered at one
        absolute position must be found and returned (with the correct
        p_src/position_offset) when the same content later appears at a
        different offset.
        """
        provider = _provider()
        content = list(range(2000, 2000 + 200))  # well past SINK_TOKENS
        donor_tokens = [0] * SINK_TOKENS + content
        kv = torch.arange(len(donor_tokens))

        ok = provider.cache_on_request_finished(
            request=_fake_req("donor-1"),
            token_ids=donor_tokens,
            kv_cache=kv,
            cache_start_pos=0,
            cache_end_pos=len(donor_tokens),
        )
        self.assertTrue(ok)
        provider.on_donor_inserted(_fake_req("donor-1"), donor_last_node_id=42)

        # Same content, arriving at a different offset in a new prompt.
        already_matched_len = 500
        prompt = list(range(9999, 9999 + already_matched_len)) + content
        result = provider.match_on_prefix_miss(
            prompt_token_ids=prompt,
            already_matched_len=already_matched_len,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.cached_start_pos, SINK_TOKENS)
        self.assertEqual(result.position_offset, already_matched_len - SINK_TOKENS)
        self.assertEqual(result.donor_last_node_id, 42)
        self.assertEqual(result.cached_token_ids, content[: result.cached_token_count])

    def test_sink_region_never_registered(self):
        """Content whose original occurrence started inside the
        attention-sink zone must never be registered as a donor —
        IRMINSUL_THEORY.md Section 7's carve-out. A regression here would
        silently start serving sink-adjacent content as if it were a
        reliable, content-only signal.
        """
        provider = _provider()
        # Entirely within the sink zone.
        donor_tokens = list(range(SINK_TOKENS))
        kv = torch.arange(len(donor_tokens))

        ok = provider.cache_on_request_finished(
            request=_fake_req("donor-sink"),
            token_ids=donor_tokens,
            kv_cache=kv,
            cache_start_pos=0,
            cache_end_pos=len(donor_tokens),
        )
        self.assertFalse(ok)
        self.assertEqual(len(provider._store), 0)

    def test_cross_tenant_isolation(self):
        """Identical content registered under one tenant's extra_key must
        never match a lookup under a different tenant's extra_key —
        PIC_IRMINSUL.md Section 2's mandatory multi-tenant isolation
        requirement. A regression here is a real cross-tenant KV leak.
        """
        provider = _provider()
        content = list(range(3000, 3000 + 200))
        donor_tokens = [0] * SINK_TOKENS + content
        kv = torch.arange(len(donor_tokens))

        provider.cache_on_request_finished(
            request=_fake_req("donor-2", extra_key="tenant_a"),
            token_ids=donor_tokens,
            kv_cache=kv,
            cache_start_pos=0,
            cache_end_pos=len(donor_tokens),
        )

        already_matched_len = 10
        prompt = list(range(10)) + content
        result = provider.match_on_prefix_miss(
            prompt_token_ids=prompt,
            already_matched_len=already_matched_len,
            extra_key="tenant_b",
        )
        self.assertIsNone(result)

        # Sanity: the same lookup under the correct tenant does match.
        result_same_tenant = provider.match_on_prefix_miss(
            prompt_token_ids=prompt,
            already_matched_len=already_matched_len,
            extra_key="tenant_a",
        )
        self.assertIsNotNone(result_same_tenant)

    def test_hash_collision_falls_back_to_miss(self):
        """Never trust the hash alone: two different chunks forced to share
        a fingerprint must not produce a match — the mandatory token-ID
        equality check is what actually guards correctness here, not the
        fingerprint's collision odds.
        """
        provider = _provider()
        donor_content = [111] * 200
        query_content = [222] * 200  # different content, forced same fingerprint

        def fake_chunks(tokens):
            return [
                Chunk(start=0, end=len(tokens), token_ids=list(tokens), fingerprint=1)
            ]

        with patch(
            "sglang.srt.mem_cache.fuzzy_match.exact_hash_provider.chunk_tokens",
            side_effect=fake_chunks,
        ):
            donor_tokens = [0] * SINK_TOKENS + donor_content
            provider.cache_on_request_finished(
                request=_fake_req("donor-3"),
                token_ids=donor_tokens,
                kv_cache=torch.arange(len(donor_tokens)),
                cache_start_pos=0,
                cache_end_pos=len(donor_tokens),
            )

            result = provider.match_on_prefix_miss(
                prompt_token_ids=[0] * 10 + query_content,
                already_matched_len=10,
            )
        self.assertIsNone(
            result,
            "fingerprint collided by construction but content differs — "
            "must fall back to miss, not serve the wrong donor",
        )


def _fixed_chunks(chunk_len):
    """``chunk_tokens`` replacement that splits into fixed-size chunks,
    fingerprinted by content so identical chunks match and different ones
    don't -- gives deterministic, controllable boundaries instead of
    depending on real CDC randomness, so multi-chunk extension logic can be
    tested precisely."""

    def _chunker(tokens):
        tokens = list(tokens)
        chunks = []
        for start in range(0, len(tokens), chunk_len):
            end = min(start + chunk_len, len(tokens))
            piece = tokens[start:end]
            chunks.append(
                Chunk(
                    start=start,
                    end=end,
                    token_ids=piece,
                    fingerprint=hash(tuple(piece)),
                )
            )
        return chunks

    return _chunker


class TestExactHashProviderMultiChunk(CustomTestCase):
    """``match_on_prefix_miss`` extends a match through consecutive chunks
    of the unmatched tail as long as they keep hitting the *same* donor's
    own contiguous position run -- see the module docstring for why that
    contiguity requirement (not true N:M segments) is what lets a combined
    match reuse ``_realize_contiguous`` unchanged. Each case here guards a
    distinct way that extension logic could silently misbehave."""

    CHUNK_LEN = 100

    def test_extends_through_multiple_hits_from_the_same_donor(self):
        """The actual new capability: a donor span spanning several CDC
        chunks must be recoverable in full, not just its first chunk --
        this is what removes the previous ~512-token single-chunk cap for
        the common case of one large shared span reused wholesale."""
        provider = _provider()
        content = list(range(5000, 5000 + 3 * self.CHUNK_LEN))  # 3 chunks
        donor_tokens = [0] * SINK_TOKENS + content

        with patch(
            "sglang.srt.mem_cache.fuzzy_match.exact_hash_provider.chunk_tokens",
            side_effect=_fixed_chunks(self.CHUNK_LEN),
        ):
            provider.cache_on_request_finished(
                request=_fake_req("donor-multi"),
                token_ids=donor_tokens,
                kv_cache=torch.arange(len(donor_tokens)),
                cache_start_pos=0,
                cache_end_pos=len(donor_tokens),
            )
            provider.on_donor_inserted(_fake_req("donor-multi"), donor_last_node_id=7)

            already_matched_len = 50
            prompt = list(range(9000, 9000 + already_matched_len)) + content
            result = provider.match_on_prefix_miss(
                prompt_token_ids=prompt,
                already_matched_len=already_matched_len,
            )

        self.assertIsNotNone(result)
        self.assertEqual(
            result.cached_token_count,
            len(content),
            "should recover all 3 chunks, not just the first",
        )
        self.assertEqual(result.cached_start_pos, SINK_TOKENS)
        self.assertEqual(result.donor_last_node_id, 7)
        self.assertEqual(result.kv_cache_indices.numel(), len(content))

    def test_stops_at_first_miss_does_not_skip_ahead(self):
        """A miss in the middle of the tail must stop the extension there
        -- must not skip the miss and pick up a later chunk that happens
        to match something else, which would silently misrepresent the
        recovered span's true (position-contiguous) donor source."""
        provider = _provider()
        chunk0_content = list(range(6000, 6000 + self.CHUNK_LEN))
        chunk1_content = list(range(7000, 7000 + self.CHUNK_LEN))  # never registered
        chunk2_content = list(range(8000, 8000 + self.CHUNK_LEN))

        with patch(
            "sglang.srt.mem_cache.fuzzy_match.exact_hash_provider.chunk_tokens",
            side_effect=_fixed_chunks(self.CHUNK_LEN),
        ):
            provider.cache_on_request_finished(
                request=_fake_req("donor-a"),
                token_ids=[0] * SINK_TOKENS + chunk0_content,
                kv_cache=torch.arange(SINK_TOKENS + self.CHUNK_LEN),
                cache_start_pos=0,
                cache_end_pos=SINK_TOKENS + self.CHUNK_LEN,
            )
            provider.on_donor_inserted(_fake_req("donor-a"), donor_last_node_id=1)
            # chunk2's content is registered too, under an unrelated donor --
            # proves it's genuinely skipped, not merely never-registered.
            provider.cache_on_request_finished(
                request=_fake_req("donor-c"),
                token_ids=[0] * SINK_TOKENS + chunk2_content,
                kv_cache=torch.arange(SINK_TOKENS + self.CHUNK_LEN),
                cache_start_pos=0,
                cache_end_pos=SINK_TOKENS + self.CHUNK_LEN,
            )
            provider.on_donor_inserted(_fake_req("donor-c"), donor_last_node_id=3)

            already_matched_len = 50
            prompt = (
                list(range(9000, 9000 + already_matched_len))
                + chunk0_content
                + chunk1_content
                + chunk2_content
            )
            result = provider.match_on_prefix_miss(
                prompt_token_ids=prompt,
                already_matched_len=already_matched_len,
            )

        self.assertIsNotNone(result)
        self.assertEqual(
            result.cached_token_count,
            self.CHUNK_LEN,
            "must stop at chunk1's miss, not continue into chunk2's separate match",
        )
        self.assertEqual(result.donor_last_node_id, 1)

    def test_stops_at_a_different_donor(self):
        """Two chunks that are each individually a valid match, but from
        *different* donor requests, must not be merged into one combined
        match -- this architecture pins a single donor node per match
        (`fuzzy_radix_cache.py`'s `donor_last_node_id`), so silently
        stitching two donors together would leave the second donor's KV
        unprotected against concurrent eviction."""
        provider = _provider()
        chunk0_content = list(range(6000, 6000 + self.CHUNK_LEN))
        chunk1_content = list(range(7000, 7000 + self.CHUNK_LEN))

        with patch(
            "sglang.srt.mem_cache.fuzzy_match.exact_hash_provider.chunk_tokens",
            side_effect=_fixed_chunks(self.CHUNK_LEN),
        ):
            provider.cache_on_request_finished(
                request=_fake_req("donor-a"),
                token_ids=[0] * SINK_TOKENS + chunk0_content,
                kv_cache=torch.arange(SINK_TOKENS + self.CHUNK_LEN),
                cache_start_pos=0,
                cache_end_pos=SINK_TOKENS + self.CHUNK_LEN,
            )
            provider.on_donor_inserted(_fake_req("donor-a"), donor_last_node_id=1)
            provider.cache_on_request_finished(
                request=_fake_req("donor-b"),
                token_ids=[0] * SINK_TOKENS + chunk1_content,
                kv_cache=torch.arange(SINK_TOKENS + self.CHUNK_LEN),
                cache_start_pos=0,
                cache_end_pos=SINK_TOKENS + self.CHUNK_LEN,
            )
            provider.on_donor_inserted(_fake_req("donor-b"), donor_last_node_id=2)

            already_matched_len = 50
            prompt = (
                list(range(9000, 9000 + already_matched_len))
                + chunk0_content
                + chunk1_content
            )
            result = provider.match_on_prefix_miss(
                prompt_token_ids=prompt,
                already_matched_len=already_matched_len,
            )

        self.assertIsNotNone(result)
        self.assertEqual(
            result.cached_token_count,
            self.CHUNK_LEN,
            "must not merge chunks from two different donor requests",
        )
        self.assertEqual(result.donor_last_node_id, 1)

    def test_stops_at_noncontiguous_position_even_with_the_same_donor_id(self):
        """A shared ``donor_last_node_id`` alone is not sufficient to merge
        two chunks -- their donor positions must also be contiguous. Guards
        against a stale/reused TreeNode id (plausible after eviction churn
        in a long-running server) coincidentally matching a genuinely
        unrelated, non-contiguous registration under the same numeric id.
        """
        provider = _provider()
        chunk0_tokens = list(range(6000, 6000 + self.CHUNK_LEN))
        chunk1_tokens = list(range(7000, 7000 + self.CHUNK_LEN))

        # Hand-construct store entries: same donor_last_node_id, but the
        # second entry's start_pos does not pick up where the first ends.
        provider._store[(None, hash(tuple(chunk0_tokens)))] = [
            _ChunkEntry(
                token_ids=chunk0_tokens,
                kv_indices=torch.arange(self.CHUNK_LEN),
                start_pos=SINK_TOKENS,
                donor_last_node_id=5,
            )
        ]
        provider._store[(None, hash(tuple(chunk1_tokens)))] = [
            _ChunkEntry(
                token_ids=chunk1_tokens,
                kv_indices=torch.arange(self.CHUNK_LEN),
                start_pos=SINK_TOKENS + self.CHUNK_LEN + 999,  # gap, same donor id
                donor_last_node_id=5,
            )
        ]

        with patch(
            "sglang.srt.mem_cache.fuzzy_match.exact_hash_provider.chunk_tokens",
            side_effect=_fixed_chunks(self.CHUNK_LEN),
        ):
            already_matched_len = 50
            prompt = (
                list(range(9000, 9000 + already_matched_len))
                + chunk0_tokens
                + chunk1_tokens
            )
            result = provider.match_on_prefix_miss(
                prompt_token_ids=prompt,
                already_matched_len=already_matched_len,
            )

        self.assertIsNotNone(result)
        self.assertEqual(
            result.cached_token_count,
            self.CHUNK_LEN,
            "same donor_last_node_id but a position gap must not extend the match",
        )


if __name__ == "__main__":
    unittest.main()
