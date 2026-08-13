"""Demonstrates, with real cached-token counts from a live server, the exact
gap PIC_IRMINSUL.md exists to close: plain RadixCache's exact-prefix
matching is strictly offset-bound, so byte-identical content at a different
absolute position is a full miss — while ExactHashProvider's content-defined
chunking finds it anyway. Not a CI test; run manually against an
already-running server:

    python test/manual/fuzzy_match/demonstrate_shift_miss_vs_hit.py <base_url>

Scenario (identical to test_exact_hash_shifted_offset_kl.py's construction —
see that file's module docstring for why the sink-prefix/pre-cached-prefix
setup is deliberate): register DONOR_TOKENS of content at absolute position
SINK_TOKENS, then request the *same* bytes again at absolute position
QUERY_PREFIX_TOKENS instead. Reports `meta_info["cached_tokens"]` from the
raw HTTP response for that second request -- no server-log scraping needed,
so this works identically against either backend:

    # Baseline: plain RadixCache, no fuzzy match -- expect cached_tokens == QUERY_PREFIX_TOKENS
    # (only the unrelated prefix matches; the donor span is a full miss despite
    # being byte-identical to already-cached content)
    sglang serve --model-path Qwen/Qwen2.5-7B-Instruct-AWQ --host 127.0.0.1 --port 21010
    python test/manual/fuzzy_match/demonstrate_shift_miss_vs_hit.py http://127.0.0.1:21010

    # With ExactHash: expect cached_tokens > QUERY_PREFIX_TOKENS -- the donor
    # span's first chunk (Stage 1's "chunks[0] only" limit) gets pulled out of
    # the miss bucket via RoPE-corrected KV reuse, not recomputed from scratch
    sglang serve --model-path Qwen/Qwen2.5-7B-Instruct-AWQ \\
      --radix-cache-backend fuzzy_match --fuzzy-match-provider ExactHash \\
      --host 127.0.0.1 --port 21000
    python test/manual/fuzzy_match/demonstrate_shift_miss_vs_hit.py http://127.0.0.1:21000
"""

import random
import sys

sys.path.insert(0, "/home/karthik/sglang-private/python")

from sglang.test.kl_test_utils import _flush_cache, _generate

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:21000"
SINK_TOKENS = 32
DONOR_TOKENS = 400
QUERY_PREFIX_TOKENS = 200
LOW, HIGH = 1000, 140000
SEED = 20260726


def rand_tokens(rng, n):
    return [rng.randint(LOW, HIGH) for _ in range(n)]


def main():
    rng = random.Random(SEED)
    donor_content = rand_tokens(rng, DONOR_TOKENS)
    query_prefix = rand_tokens(rng, QUERY_PREFIX_TOKENS)

    _flush_cache(BASE_URL)

    # Register donor_content at absolute position [SINK_TOKENS, SINK_TOKENS+DONOR_TOKENS).
    sink_prefix = list(range(50000, 50000 + SINK_TOKENS))
    _generate(BASE_URL, [sink_prefix + donor_content], max_new_tokens=0)

    # Pre-cache query_prefix standalone, so the combined request below gets a
    # full exact match on it -- isolates the donor span as the entire
    # unmatched tail, chunked fresh from its own start.
    _generate(BASE_URL, [query_prefix], max_new_tokens=0)

    # Request the *same* donor_content bytes again, now at absolute position
    # [QUERY_PREFIX_TOKENS, QUERY_PREFIX_TOKENS+DONOR_TOKENS) -- a shifted
    # offset, identical content.
    results = _generate(
        BASE_URL, [query_prefix + donor_content], max_new_tokens=1
    )
    cached_tokens = results[0]["meta_info"]["cached_tokens"]
    total_tokens = QUERY_PREFIX_TOKENS + DONOR_TOKENS

    print(f"cached_tokens={cached_tokens} / total={total_tokens}")
    if cached_tokens <= QUERY_PREFIX_TOKENS:
        print(
            f"MISS on the shifted donor span: only the {QUERY_PREFIX_TOKENS}-token "
            "prefix matched -- plain exact-prefix matching cannot see identical "
            "content at a different absolute position."
        )
    else:
        recovered = cached_tokens - QUERY_PREFIX_TOKENS
        print(
            f"HIT: {recovered} of {DONOR_TOKENS} donor tokens recovered from the "
            "shifted-offset span (Stage 1's chunker only matches the first chunk "
            "of a longer unmatched tail, so recovered < DONOR_TOKENS is expected)."
        )


if __name__ == "__main__":
    main()
