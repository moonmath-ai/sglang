"""Regression check for the divergence-aligned donor registration fix.

Bug this guards (found 2026-08-06, `stage2_reproduction.md` Part B): a content
block shared by two requests was unreusable whenever each request reached it
through *its own* cached preamble -- i.e. the realistic multi-tenant shape. On
the pre-fix code this script recovered **0 of 2000** tokens; after the fix it
recovers **1999 of 2000**.

Mechanism: `ExactHashProvider` can only reuse a block starting at the recipient's
first unmatched token (`_match_contiguous_run` stops at the first chunk that
misses, and the prefix-shaped `device_indices` contract cannot represent a hole),
so the recipient's leading chunk is always cut from its own divergence point.
Registration, however, chunked only from `SINK_TOKENS`; for a donor that had a
cached prefix, that puts its chunk boundaries *inside* the preamble, so the chunk
covering the block's start straddles the boundary and never equals the
recipient's leading chunk. `FuzzyRadixCache._register_divergence_aligned` adds a
second registration pass chunked from the donor's own divergence point, which
produces exactly the alignment such a recipient will produce.

Run against a live fuzzy_match server:

    python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \
      --radix-cache-backend fuzzy_match --fuzzy-match-provider ExactHash \
      --trust-remote-code --attention-backend flashinfer --port 21000
    python test/manual/fuzzy_match/verify_divergence_aligned_registration.py

Exits non-zero if the block is not reused, so it can gate a change.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))

from sglang.test.kl_test_utils import _flush_cache, _generate

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:21000"

SHARED_BLOCK_TOKENS = 2000
TENANT_A_PREAMBLE = 500
TENANT_B_PREAMBLE = 300  # deliberately a different length from A's
LOW, HIGH = 1000, 99000  # DeepSeek-V2-Lite-Chat vocab_size=102400
SEED = 4242
# The block is 2000 tokens; a partial recovery still proves the alignment works,
# so gate well above "nothing" rather than demanding every token.
MIN_EXPECTED_RECOVERY = 1000


def rand_tokens(rng, n):
    return [rng.randint(LOW, HIGH) for _ in range(n)]


def main():
    rng = random.Random(SEED)
    shared_block = rand_tokens(rng, SHARED_BLOCK_TOKENS)
    preamble_a = rand_tokens(rng, TENANT_A_PREAMBLE)
    preamble_b = rand_tokens(rng, TENANT_B_PREAMBLE)

    _flush_cache(BASE_URL)

    # Tenant A warms its own preamble, then sends the request carrying the
    # shared block -- so A's *new* content begins exactly at the block.
    _generate(BASE_URL, [preamble_a], max_new_tokens=0)
    _generate(BASE_URL, [preamble_a + shared_block], max_new_tokens=0)

    # Tenant B warms its own (different-length) preamble, then asks for the
    # same block. B's unmatched tail also begins exactly at the block, but at a
    # different absolute offset -- unreachable for exact-prefix matching.
    _generate(BASE_URL, [preamble_b], max_new_tokens=0)
    results = _generate(BASE_URL, [preamble_b + shared_block], max_new_tokens=1)

    cached_tokens = results[0]["meta_info"]["cached_tokens"]
    recovered = max(0, cached_tokens - TENANT_B_PREAMBLE)
    print(
        f"tenant B [{TENANT_B_PREAMBLE}]+[{SHARED_BLOCK_TOKENS}] "
        f"cached_tokens={cached_tokens} -> PIC recovered "
        f"{recovered}/{SHARED_BLOCK_TOKENS}"
    )

    if recovered >= MIN_EXPECTED_RECOVERY:
        print("PASS: shared block reused across tenants at a shifted offset.")
        return 0
    print(
        "FAIL: no cross-tenant reuse. Donor registration is not aligned to the "
        "divergence point (this is the pre-fix behaviour: 0 tokens recovered)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
