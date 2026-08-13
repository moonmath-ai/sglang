"""Part B: RAG-shaped agentic traffic — a retrieved document reappearing across
conversations at different absolute offsets.

This is the shape `stage2_reproduction.md` Part B flagged as PIC's best case and
left untested. Unlike the BFCL tool-schema arm (where reuse turns out to be
prefix-aligned and plain RadixCache already gets it), here the same retrieved
document genuinely lands at a *different* absolute offset in every conversation,
because each conversation has accumulated a different amount of transcript.

Why the layout is append-only, and why that matters
---------------------------------------------------
`ExactHashProvider` can only reuse a block that begins at the recipient's *first
unmatched token* (`_match_contiguous_run` stops at the first chunk that misses,
and the prefix-shaped `device_indices` contract cannot represent a hole). So the
retrieved document has to be the first new content in the request, or nothing
matches.

An **append-only** transcript gives exactly that, for free:

    prompt(turn t) = SYSTEM + sum_{i<t}(DOC_i + query_i + answer_i) + DOC_t + query_t

Turn t's prefix is precisely turn t-1's prompt plus its answer, which is already
cached, so the exact-prefix match consumes everything up to `DOC_t` and `DOC_t`
begins the unmatched tail. Across conversations the preceding transcript differs
in length, so `DOC_t` sits at a different offset each time -- invisible to
exact-prefix matching, reusable by PIC.

**If instead a deployment drops previously-retrieved documents from the
transcript** (re-retrieving fresh context each turn), the tail begins with the
previous turn's query/answer and the document sits behind ~100 tokens of new
text -- which recovers nothing (measured: 137 tokens of lead-in takes recovery
from 1999/2000 to 0). That distinction is a deployment-shape finding, not a
tuning knob: run `--drop-history-docs` to measure it.

Corpus: documents are real prose slices from wikitext-2 (locally cached);
queries are real user turns from ShareGPT. Answers are short deterministic
filler so replays are byte-identical across arms.

Usage:
    python test/manual/fuzzy_match/measure_agentic_rag_replay.py <base_url> \
        [--conversations N] [--turns N] [--doc-tokens N] [--drop-history-docs] \
        [--out results.json] [--label NAME]

    python test/manual/fuzzy_match/measure_agentic_rag_replay.py --compare a.json b.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import requests

sys.path.insert(0, "/home/karthik/sglang-private/python")

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"

SYSTEM = (
    "You are a research assistant. Answer the user's question using the "
    "retrieved passages provided with each turn. Cite the passage you used.\n\n"
)
ANSWER_FILLER = " Based on the retrieved passage, here is the answer.\n\n"
MAX_PROMPT_TOKENS = 15000


def load_documents(tokenizer, num_docs, doc_tokens):
    """Real prose slices from wikitext-2, sized like RAG retrieval chunks."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    # Skip headers/blank lines; wikitext headings look like " = Title = ".
    text = " ".join(
        line.strip()
        for line in ds["text"]
        if len(line.strip()) > 200 and not line.strip().startswith("=")
    )
    ids = tokenizer(text, add_special_tokens=False).input_ids
    docs = []
    for i in range(num_docs):
        chunk = ids[i * doc_tokens : (i + 1) * doc_tokens]
        header = tokenizer(
            f"\n[Retrieved passage {i}]\n", add_special_tokens=False
        ).input_ids
        docs.append(header + chunk)
    return docs


def load_queries(tokenizer, count):
    """Real user turns from ShareGPT, used as the per-turn question."""
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train",
    )
    out = []
    for row in ds:
        for msg in row.get("conversations", []):
            if msg.get("from") == "human":
                value = msg.get("value", "").strip()
                if 40 < len(value) < 300:
                    out.append(
                        tokenizer(
                            f"\nQuestion: {value}\nAnswer:", add_special_tokens=False
                        ).input_ids
                    )
                    break
        if len(out) >= count:
            break
    return out


def build_conversation(conv_idx, docs, queries, tokenizer, turns, drop_history_docs):
    """Token-id prompts for one conversation's turns, transcript growing per turn.

    Each conversation retrieves the shared documents in a *rotated* order, so a
    given document appears after a different amount of transcript in every
    conversation -- that offset spread is the whole point.
    """
    system = tokenizer(SYSTEM, add_special_tokens=False).input_ids
    answer = tokenizer(ANSWER_FILLER, add_special_tokens=False).input_ids

    prompts, transcript = [], []
    for turn in range(turns):
        doc = docs[(conv_idx + turn) % len(docs)]
        query = queries[(conv_idx * turns + turn) % len(queries)]
        prompt = system + transcript + doc + query
        if len(prompt) > MAX_PROMPT_TOKENS:
            break
        prompts.append({"turn": turn, "doc": (conv_idx + turn) % len(docs), "input_ids": prompt})
        # Append-only keeps the retrieved doc in the transcript, so the next
        # turn's prefix is exactly this prompt + answer and the next document
        # lands at the head of the unmatched tail. Dropping it instead puts the
        # previous query/answer in front of the next document.
        if drop_history_docs:
            transcript = transcript + query + answer
        else:
            transcript = transcript + doc + query + answer
    return prompts


def _run_conversation(base_url, docs, queries, tokenizer, conv_idx, turns, drop_history_docs):
    """One conversation's turns, strictly in order (the append-only dependency:
    turn t's prefix is turn t-1's prompt + answer)."""
    records = []
    for req in build_conversation(
        conv_idx, docs, queries, tokenizer, turns, drop_history_docs
    ):
        t0 = time.perf_counter()
        resp = requests.post(
            base_url + "/generate",
            json={
                "input_ids": req["input_ids"],
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
            },
            timeout=600,
        )
        latency_ms = (time.perf_counter() - t0) * 1e3
        meta = resp.json()["meta_info"]
        records.append(
            {
                "conversation": conv_idx,
                "turn": req["turn"],
                "doc": req["doc"],
                "prompt_tokens": meta["prompt_tokens"],
                "cached_tokens": meta["cached_tokens"],
                "latency_ms": latency_ms,
            }
        )
    return records


def replay(base_url, docs, queries, tokenizer, conversations, turns, drop_history_docs, concurrency=1):
    if concurrency > 1:
        # Conversations in parallel (they are independent), turns within each
        # conversation still sequential. NOTE: cross-conversation PIC hits
        # depend on another conversation having registered the doc first, so
        # under concurrency the hit/miss mix is interleaving-dependent and
        # recovery is no longer directly comparable to the sequential arm.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            results = list(
                ex.map(
                    lambda c: _run_conversation(
                        base_url, docs, queries, tokenizer, c, turns, drop_history_docs
                    ),
                    range(conversations),
                )
            )
        records = [r for recs in results for r in recs]
        print(f"  ... {conversations}/{conversations} conversations, "
              f"{len(records)} requests (concurrency={concurrency})")
        return records

    records = []
    for conv_idx in range(conversations):
        records.extend(
            _run_conversation(
                base_url, docs, queries, tokenizer, conv_idx, turns, drop_history_docs
            )
        )
        if (conv_idx + 1) % 8 == 0:
            print(f"  ... {conv_idx + 1}/{conversations} conversations, "
                  f"{len(records)} requests")
    return records


def summarize(label, records, wall_time_s=None):
    prompt_total = sum(r["prompt_tokens"] for r in records)
    cached_total = sum(r["cached_tokens"] for r in records)
    lat = [r["latency_ms"] for r in records]
    out = {
        "label": label,
        "requests": len(records),
        "prompt_tokens_total": prompt_total,
        "cached_tokens_total": cached_total,
        "recovery_pct": 100.0 * cached_total / prompt_total if prompt_total else 0.0,
        "latency_mean_ms": statistics.mean(lat),
        "latency_median_ms": statistics.median(lat),
        "latency_p95_ms": sorted(lat)[max(0, int(0.95 * len(lat)) - 1)],
        "latency_total_s": sum(lat) / 1e3,
        "wall_time_s": wall_time_s,
    }
    print(f"\n=== {label} ===")
    print(f"  requests               : {out['requests']}")
    print(f"  prompt tokens (total)  : {out['prompt_tokens_total']:,}")
    print(f"  cached tokens (total)  : {out['cached_tokens_total']:,}")
    print(f"  recovery rate          : {out['recovery_pct']:.1f}%")
    print(f"  latency mean / median  : {out['latency_mean_ms']:.1f} / "
          f"{out['latency_median_ms']:.1f} ms")
    print(f"  latency p95            : {out['latency_p95_ms']:.1f} ms")
    print(f"  latency total          : {out['latency_total_s']:.2f} s")
    if wall_time_s is not None:
        # Sum-of-latencies grows with concurrency by design (batching slows
        # each request); wall time is what shows whether throughput won.
        print(f"  wall time              : {wall_time_s:.2f} s "
              f"({len(records) / wall_time_s:.1f} req/s)")
    return out


def compare(path_a, path_b):
    a, b = json.load(open(path_a)), json.load(open(path_b))
    sa, sb = a["summary"], b["summary"]
    print(f"\n{'metric':26s} {sa['label']:>18s} {sb['label']:>18s} {'delta':>12s}")
    for key, fmt in (
        ("recovery_pct", "{:.1f}%"),
        ("latency_mean_ms", "{:.1f}ms"),
        ("latency_median_ms", "{:.1f}ms"),
        ("latency_total_s", "{:.2f}s"),
    ):
        va, vb = sa[key], sb[key]
        print(f"{key:26s} {fmt.format(va):>18s} {fmt.format(vb):>18s} "
              f"{(vb - va):>+12.2f}")
    if sb["latency_total_s"]:
        print(f"\nend-to-end speedup ({sa['label']} / {sb['label']}): "
              f"{sa['latency_total_s'] / sb['latency_total_s']:.2f}x")

    ra = {(r["conversation"], r["turn"]): r for r in a["records"]}
    pairs = [
        (ra[k]["latency_ms"], r["latency_ms"], r["cached_tokens"] - ra[k]["cached_tokens"])
        for r in b["records"]
        if (k := (r["conversation"], r["turn"])) in ra
    ]
    if pairs:
        wins = sum(1 for x, y, _ in pairs if y < x)
        extra = [d for _, _, d in pairs]
        print(f"paired requests: {len(pairs)}, arm-B faster on {wins} "
              f"({100.0 * wins / len(pairs):.0f}%)")
        print(f"extra cached tokens per request (arm B - arm A): "
              f"mean={statistics.mean(extra):.0f} max={max(extra)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url", nargs="?", default="http://127.0.0.1:21000")
    ap.add_argument("--conversations", type=int, default=24)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--doc-tokens", type=int, default=1500)
    ap.add_argument("--num-docs", type=int, default=6)
    ap.add_argument(
        "--drop-history-docs",
        action="store_true",
        help="drop previously-retrieved docs from the transcript (re-retrieval "
        "shape); puts the previous query/answer in front of the new document",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="concurrent client workers: this many conversations in flight at "
        "once (turns within each conversation stay sequential). Default 1 = "
        "fully sequential replay. Under concurrency the PIC hit/miss mix is "
        "interleaving-dependent; recovery is not comparable to the sequential arm",
    )
    ap.add_argument("--out")
    ap.add_argument("--label", default="rag")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    docs = load_documents(tokenizer, args.num_docs, args.doc_tokens)
    queries = load_queries(tokenizer, args.conversations * args.turns)
    print(f"documents: {len(docs)} x ~{args.doc_tokens} tokens "
          f"(actual {[len(d) for d in docs[:3]]}...)")
    print(f"queries: {len(queries)}, conversations: {args.conversations} x "
          f"{args.turns} turns, "
          f"layout={'re-retrieval' if args.drop_history_docs else 'append-only'}")

    print(f"\nreplaying against {args.base_url} ... (concurrency={args.concurrency})")
    t_replay_start = time.perf_counter()
    records = replay(
        args.base_url,
        docs,
        queries,
        tokenizer,
        args.conversations,
        args.turns,
        args.drop_history_docs,
        args.concurrency,
    )
    wall_time_s = time.perf_counter() - t_replay_start
    summary = summarize(args.label, records, wall_time_s)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "records": records}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
