"""Phase 0-style recovery-rate measurement (PIC_IRMINSUL.md Section 2/3
Measurements) — a test-case experiment on a real (adapted) corpus, not a
production-traffic prediction. Not a CI test; run manually:

    python test/manual/fuzzy_match/measure_recovery_rate.py

Methodology
-----------
LongBench-v2 has real document reuse: many entries share the same `context`
across multiple different `question`s (confirmed directly — in a 400-example
scan, 11 groups shared a context across 2-6 questions each). But the raw
dataset's own prompt shape (`context + " " + question`, context always
first) means ordinary exact-prefix RadixCache matching *already* recovers
that overlap for free — the shared content sits at the same absolute
position (0) every time. That doesn't exercise what this mechanism adds.

To measure the position-*shift*-specific recovery this mechanism targets
(PIC_IRMINSUL.md Section 2's Phase 0 distinction between fleet-locality
misses and genuine position-shift misses), each repeated use of a shared
context is prepended with a synthetic "prior turn" of random, unique length
— simulating the same document appearing after a varying amount of prior
conversation/tool-call history, which is exactly the agentic pattern this
whole plan is about. This is an adapted corpus, not raw LongBench-v2 — that
adaptation is deliberate and documented, not hidden.

Simulation, per shared-context group, processing requests in arrival order:
  - request 1 (first use of this context): nothing to recover yet; register
    its chunks.
  - request 2..K: exact_matched_len is 0 by construction (each prior-turn is
    unique, so ordinary exact-prefix matching never fires) — chunk the
    entire prompt and check how many tokens' worth of chunks already exist
    in the registry from earlier requests. That's the recovery rate PIC
    itself is meant to add. Then register this request's own new chunks.
"""

import hashlib
import random
import sys
from collections import defaultdict

sys.path.insert(0, "/home/karthik/sglang-private/python")

from datasets import load_dataset
from transformers import AutoTokenizer

from sglang.srt.mem_cache.fuzzy_match.chunker import chunk_tokens

MODEL = "moonshotai/Moonlight-16B-A3B-Instruct"
SCAN_EXAMPLES = 2000
MIN_GROUP_SIZE = 2
PRIOR_TURN_MIN, PRIOR_TURN_MAX = 0, 800
SEED = 20260726


def load_context_groups():
    ds = load_dataset("THUDM/LongBench-v2", split="train", streaming=True)
    groups = defaultdict(list)
    for i, ex in enumerate(ds):
        if i >= SCAN_EXAMPLES:
            break
        h = hashlib.md5(ex["context"].encode()).hexdigest()
        groups[h].append(ex)
    return {h: exs for h, exs in groups.items() if len(exs) >= MIN_GROUP_SIZE}


def main():
    rng = random.Random(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    groups = load_context_groups()
    print(
        f"Scanned {SCAN_EXAMPLES} examples, found {len(groups)} context groups "
        f"with >= {MIN_GROUP_SIZE} questions each "
        f"(sizes: {sorted((len(v) for v in groups.values()), reverse=True)[:15]})"
    )

    registry = set()  # fingerprints of all previously-registered chunks
    per_request_recovery = []  # excludes each group's first request
    per_request_ceiling = []  # theoretical max given the corpus shape
    hit_count = 0
    total_non_first_requests = 0

    for h, examples in groups.items():
        for idx, ex in enumerate(examples):
            context_tokens = tokenizer.encode(ex["context"])
            question_tokens = tokenizer.encode(ex["question"])
            prior_turn = [
                rng.randint(1000, 140000)
                for _ in range(rng.randint(PRIOR_TURN_MIN, PRIOR_TURN_MAX))
            ]
            prompt = prior_turn + context_tokens + question_tokens

            chunks = chunk_tokens(prompt)
            if idx > 0:
                total_non_first_requests += 1
                recoverable = sum(
                    len(c.token_ids) for c in chunks if c.fingerprint in registry
                )
                rate = recoverable / len(prompt) if prompt else 0.0
                per_request_recovery.append(rate)
                # Only the shared context portion can ever be recoverable —
                # prior_turn and question are unique per request by
                # construction. This is the ceiling THIS corpus shape
                # imposes, not a universal number.
                per_request_ceiling.append(len(context_tokens) / len(prompt))
                if recoverable > 0:
                    hit_count += 1

            for c in chunks:
                registry.add(c.fingerprint)

    if not per_request_recovery:
        print("No non-first requests found — check MIN_GROUP_SIZE / SCAN_EXAMPLES.")
        return

    n = len(per_request_recovery)
    avg = sum(per_request_recovery) / n
    avg_ceiling = sum(per_request_ceiling) / n
    # Recovery relative to what this corpus shape could ever yield, not
    # relative to the full prompt — this is the number that actually
    # isolates mechanism effectiveness from corpus-shape ceiling effects.
    avg_of_ceiling = sum(
        r / c for r, c in zip(per_request_recovery, per_request_ceiling) if c > 0
    ) / n

    per_request_recovery_sorted = sorted(per_request_recovery)
    print(f"\nNon-first requests measured: {n}")
    print(f"Requests with any recoverable content: {hit_count}/{n} "
          f"({100 * hit_count / n:.1f}%)")
    print(f"Average recovery rate (of full prompt): {100 * avg:.1f}%")
    print(f"Average theoretical ceiling (context / prompt length): {100 * avg_ceiling:.1f}%")
    print(f"Average recovery AS A FRACTION OF THE CEILING: {100 * avg_of_ceiling:.1f}%")
    print(
        "Recovery-rate distribution, of full prompt (min/p25/p50/p75/max): "
        + ", ".join(
            f"{100 * per_request_recovery_sorted[int(p * (n - 1))]:.1f}%"
            for p in (0, 0.25, 0.5, 0.75, 1.0)
        )
    )


if __name__ == "__main__":
    main()
