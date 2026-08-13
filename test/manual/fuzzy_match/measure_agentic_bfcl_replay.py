"""Part B: real agentic-workload measurement via BFCL v3 multi-turn replay.

`stage2_reproduction.md` Part B asks for an external agentic test whose one
required property is that **the same content block reappears byte-identical at a
different absolute token offset across requests**. This script gets that property
*from the corpus itself* rather than by synthetic adaptation, which is what makes
it a stronger test than `measure_recovery_rate_mla.py` (T4, which prepends
random filler to force the shift).

Where the offset shift comes from
---------------------------------
BFCL v3's `multi_turn_base` sessions each declare `involved_classes` -- the API
surfaces that session's agent has access to -- and the *combinations and orders
genuinely vary* across sessions (`('TicketAPI','TravelAPI')`,
`('TwitterAPI','VehicleControlAPI')`, ...). Each API's schema document is
therefore preceded by a different amount of other schema text depending on which
session is running, so `TravelAPI`'s block sits at many different absolute
offsets across the workload. Plain RadixCache's exact-prefix matching cannot see
that reuse; PIC's content-defined chunking can. Growing per-turn conversation
history shifts things further within a session.

No synthetic prior-turn padding is used. The only synthesized text is short
deterministic assistant/tool filler between user turns, needed so conversation
history grows at all -- it is *not* the repeated content being measured.

Realistic block sizes (measured, not assumed): BFCL's per-API schema documents
are 2208-6159 tokens each (9-22 functions), so a 2-API session carries roughly
4.5k-12k tokens of repeated schema per request. Note this is well *above* the
~750-1100-token PIC break-even measured in `stage2_reproduction.md` T6b --
recording it here because an earlier assumption that agent tool schemas are
"100-400 tokens" came from single-function toy corpora and was wrong.

What it measures
----------------
Runs the identical request stream against whichever server you point it at, so
the same traffic can be A/B'd across cache backends (plain RadixCache vs
`fuzzy_match`) and attention backends. Per request it records prompt tokens,
`cached_tokens`, and wall-clock latency at `max_new_tokens=1` (dominated by
prefill, so it stands in for TTFT -- same method as
`measure_conditional_ttft_mla.py`).

Usage:
    python test/manual/fuzzy_match/measure_agentic_bfcl_replay.py <base_url> \
        [--sessions N] [--out results.json] [--label NAME]

Compare two runs:
    python test/manual/fuzzy_match/measure_agentic_bfcl_replay.py --compare a.json b.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter

import requests

sys.path.insert(0, "/home/karthik/sglang-private/python")

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"
BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# BFCL declares agent capabilities as class names; the schema documents are
# filenames. This mapping is from BFCL's own repo layout.
CLASS_TO_DOC = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}

PREAMBLE = (
    "You are an autonomous agent. You have access to the following tool APIs. "
    "Call them as needed to satisfy the user's request.\n\n"
)

# Keep prompts inside the server's max_prefill_tokens (16384 by default).
MAX_APIS_PER_SESSION = 2
MAX_PROMPT_TOKENS = 15000


def _jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(tokenizer):
    """Return (schema_token_blocks, sessions) both derived from BFCL v3."""
    schemas = {}
    for cls, doc in CLASS_TO_DOC.items():
        path = hf_hub_download(
            BFCL_REPO, f"multi_turn_func_doc/{doc}.json", repo_type="dataset"
        )
        text = f"## API: {cls}\n" + json.dumps(_jsonl(path), indent=2) + "\n\n"
        schemas[cls] = tokenizer(text, add_special_tokens=False).input_ids

    path = hf_hub_download(
        BFCL_REPO, "BFCL_v3_multi_turn_base.json", repo_type="dataset"
    )
    sessions = [
        s
        for s in _jsonl(path)
        # Sessions whose APIs we have schema documents for, small enough to fit.
        if s["involved_classes"]
        and len(s["involved_classes"]) <= MAX_APIS_PER_SESSION
        and all(c in CLASS_TO_DOC for c in s["involved_classes"])
    ]
    return schemas, sessions


def build_tenant_preambles(sessions, tokenizer, num_tenants):
    """Per-tenant system preambles of *deliberately different* lengths.

    This is the multi-tenant emulation. Real multi-tenant agentic serving gives
    each tenant its own system prompt -- brand voice, policy text, user context,
    dates -- and it sits *in front of* the shared tool-schema block. Because
    those preambles differ in length, an identical tool schema lands at a
    different absolute offset for every tenant. That is precisely the
    "byte-identical content at a different absolute offset" property Part B
    requires, and it is the case ordinary exact-prefix matching structurally
    cannot recover: tenant A's cached schema is useless to tenant B despite
    being the same tokens.

    Preamble text is drawn from *held-out* BFCL user queries (sessions past the
    measured slice), so no new corpus dependency and no random filler. Lengths
    are deliberately non-round and unevenly spaced (64 + 137*i) so no tenant's
    schema offset accidentally aligns with another's or with a chunk boundary.
    """
    pool = []
    for session in sessions:
        for turn in session["question"]:
            for msg in turn:
                if msg.get("role") == "user":
                    pool.append(msg.get("content", ""))
    ids = tokenizer(" ".join(pool), add_special_tokens=False).input_ids

    preambles, offset = [], 0
    for i in range(num_tenants):
        length = 64 + 137 * i
        header = tokenizer(
            f"You are the assistant for tenant {i}. Operating policy:\n",
            add_special_tokens=False,
        ).input_ids
        # Distinct slice per tenant so preambles differ in content as well as
        # length -- two tenants must not share a prefix by accident.
        preambles.append(header + ids[offset : offset + length])
        offset += length
    return preambles


def build_session_requests(
    session, schemas, tokenizer, shuffle_apis=False, tenant_preamble=None
):
    """Token-id prompts for each turn of one session, history growing per turn.

    Pieces are tokenized separately and concatenated so a schema block is a
    byte-identical token subsequence wherever it lands -- the same construction
    the other manual scripts in this directory use.

    ``shuffle_apis`` permutes the order of this session's tool-schema blocks,
    deterministically per session id. Nothing about the *content* changes -- it
    is still 100% corpus text -- but a given API's block stops being
    prefix-aligned across sessions. This matters because the unshuffled arm
    turns out to be fully recoverable by ordinary exact-prefix matching (the
    schema head sits at offset 0 and BFCL orders APIs consistently within a
    combination), leaving PIC nothing to do. Tool-list ordering is incidental in
    a real agent, so permuting it is a realistic source of the position shift
    PIC targets -- and a much weaker assumption than synthesizing filler.
    """
    # Tenant preamble first, then the generic instruction, then the shared tool
    # schemas -- so the schema block's absolute offset is set by the tenant.
    head = list(tenant_preamble or [])
    head = head + tokenizer(PREAMBLE, add_special_tokens=False).input_ids
    classes = list(session["involved_classes"])
    if shuffle_apis:
        # Deterministic per-session permutation, so replays are identical
        # across arms and runs.
        random.Random(session["id"]).shuffle(classes)
    for cls in classes:
        head = head + schemas[cls]

    requests_out = []
    history: list[int] = []
    for turn_idx, turn in enumerate(session["question"]):
        user_text = "\n".join(
            m.get("content", "") for m in turn if m.get("role") == "user"
        )
        if not user_text:
            continue
        turn_ids = tokenizer(
            f"User: {user_text}\nAssistant:", add_special_tokens=False
        ).input_ids
        prompt = head + history + turn_ids
        if len(prompt) > MAX_PROMPT_TOKENS:
            break
        requests_out.append({"turn": turn_idx, "input_ids": prompt})
        # Deterministic filler so history grows; not the measured content.
        history = (
            history
            + turn_ids
            + tokenizer(
                f" Calling tool for step {turn_idx}. Done.\n",
                add_special_tokens=False,
            ).input_ids
        )
    return requests_out


def replay(
    base_url,
    sessions,
    schemas,
    tokenizer,
    limit,
    shuffle_apis=False,
    tenant_preambles=None,
):
    records = []
    used = sessions[:limit]
    for si, session in enumerate(used):
        # Round-robin tenants so each API combination is exercised by several
        # tenants -- that cross-tenant repetition is the PIC opportunity.
        tenant = si % len(tenant_preambles) if tenant_preambles else None
        for req in build_session_requests(
            session,
            schemas,
            tokenizer,
            shuffle_apis=shuffle_apis,
            tenant_preamble=tenant_preambles[tenant] if tenant_preambles else None,
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
                    "session": session["id"],
                    "tenant": tenant,
                    "apis": session["involved_classes"],
                    "turn": req["turn"],
                    "prompt_tokens": meta["prompt_tokens"],
                    "cached_tokens": meta["cached_tokens"],
                    "latency_ms": latency_ms,
                }
            )
        if (si + 1) % 10 == 0:
            print(f"  ... {si + 1}/{len(used)} sessions, {len(records)} requests")
    return records


def summarize(label, records):
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
        "latency_p95_ms": sorted(lat)[int(0.95 * len(lat)) - 1],
        "latency_total_s": sum(lat) / 1e3,
    }
    print(f"\n=== {label} ===")
    print(f"  requests               : {out['requests']}")
    print(f"  prompt tokens (total)  : {out['prompt_tokens_total']:,}")
    print(f"  cached tokens (total)  : {out['cached_tokens_total']:,}")
    print(f"  recovery rate          : {out['recovery_pct']:.1f}%")
    print(
        f"  latency mean / median  : {out['latency_mean_ms']:.1f} / "
        f"{out['latency_median_ms']:.1f} ms"
    )
    print(f"  latency p95            : {out['latency_p95_ms']:.1f} ms")
    print(f"  latency total          : {out['latency_total_s']:.2f} s")
    return out


def compare(path_a, path_b):
    a, b = json.load(open(path_a)), json.load(open(path_b))
    sa, sb = a["summary"], b["summary"]
    print(f"\n{'metric':26s} {sa['label']:>18s} {sb['label']:>18s} {'delta':>12s}")
    for key, fmt, better in (
        ("recovery_pct", "{:.1f}%", "higher"),
        ("latency_mean_ms", "{:.1f}ms", "lower"),
        ("latency_median_ms", "{:.1f}ms", "lower"),
        ("latency_total_s", "{:.2f}s", "lower"),
    ):
        va, vb = sa[key], sb[key]
        d = f"{(vb - va):+.2f}"
        print(f"{key:26s} {fmt.format(va):>18s} {fmt.format(vb):>18s} {d:>12s}")
    if sb["latency_total_s"]:
        print(
            f"\nend-to-end speedup ({sa['label']} / {sb['label']}): "
            f"{sa['latency_total_s'] / sb['latency_total_s']:.2f}x"
        )

    # Per-request paired deltas -- same traffic, so requests line up 1:1.
    ra = {(r["session"], r["turn"]): r for r in a["records"]}
    pairs = [
        (
            ra[k]["latency_ms"],
            r["latency_ms"],
            r["cached_tokens"] - ra[k]["cached_tokens"],
        )
        for r in b["records"]
        if (k := (r["session"], r["turn"])) in ra
    ]
    if pairs:
        wins = sum(1 for x, y, _ in pairs if y < x)
        print(
            f"paired requests: {len(pairs)}, arm-B faster on {wins} "
            f"({100.0 * wins / len(pairs):.0f}%)"
        )
        extra = [d for _, _, d in pairs]
        print(
            f"extra cached tokens per request (arm B - arm A): "
            f"mean={statistics.mean(extra):.0f} max={max(extra)}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url", nargs="?", default="http://127.0.0.1:21000")
    ap.add_argument("--sessions", type=int, default=40)
    ap.add_argument("--out")
    ap.add_argument("--label", default="run")
    ap.add_argument(
        "--shuffle-apis",
        action="store_true",
        help="permute each session's tool-block order (breaks prefix alignment)",
    )
    ap.add_argument(
        "--tenants",
        type=int,
        default=0,
        help="multi-tenant emulation: N tenants with variable-length preambles "
        "in front of the shared tool schemas (0 = single tenant, off)",
    )
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    schemas, sessions = load_corpus(tokenizer)

    tenant_preambles = None
    if args.tenants:
        # Held-out sessions only, so preamble text never overlaps measured traffic.
        tenant_preambles = build_tenant_preambles(
            sessions[args.sessions :], tokenizer, args.tenants
        )
        print(
            f"tenants: {args.tenants}, preamble lengths "
            f"{[len(p) for p in tenant_preambles]}"
        )
    print(
        f"schema blocks: "
        + ", ".join(f"{c}={len(v)}tok" for c, v in sorted(schemas.items()))
    )
    combos = Counter(tuple(s["involved_classes"]) for s in sessions[: args.sessions])
    print(
        f"eligible sessions: {len(sessions)} (using {args.sessions}), "
        f"{len(combos)} distinct API combinations"
    )

    print(f"\nreplaying against {args.base_url} ...")
    records = replay(
        args.base_url,
        sessions,
        schemas,
        tokenizer,
        args.sessions,
        shuffle_apis=args.shuffle_apis,
        tenant_preambles=tenant_preambles,
    )
    summary = summarize(args.label, records)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "records": records}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
