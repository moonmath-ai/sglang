# Fuzzy KV Cache Reuse (`--radix-cache-backend fuzzy_match`)

Position-independent KV cache reuse (Irminsul-style PIC) for MLA models.
Standard RadixAttention reuses KV only when a prefix is **byte-identical at
the same absolute offset**; this backend additionally reuses donor KV for
content that is byte-identical but sits at a **different offset** than where
it was originally computed. When exact prefix matching leaves part of a
prompt uncovered, the `ExactHashProvider` (content-defined chunking +
exact-hash, lossless) may nominate donor KV from a previously finished
request; the donor KV is position-corrected (delta-RoPE) into
recipient-owned slots before the forward pass. Reuse follows the
`|exact|fuzzy|miss|` prompt decomposition: one contiguous fuzzy span
anchored at the exact prefix boundary.

## Enabling

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \
  --radix-cache-backend fuzzy_match
```

Selecting the backend enables the feature; `ExactHash` is the only in-tree
provider. Reuse markers in the server log: `fuzzy match success` (match
accepted), `[FUZZY] Realized N fuzzy tokens` (KV copied + RoPE-corrected),
and `#fuzzy-token: N` in the prefill batch line. The fused single-launch
realization kernel is opt-in: `SGLANG_FUZZY_REALIZE_FUSED=1`.

## End-to-end call chain

One request, from arrival to donor registration. Ownership of locks,
allocation, and RoPE is called out at each step.

```
Scheduler admits request
  └─ Req.init_next_round_input                    (managers/schedule_batch.py)
       └─ FuzzyRadixCache.match_prefix(params)    (fuzzy_radix_cache.py)
            1. RadixCache.match_prefix (super)  -> exact device_indices, last_node
            2. Gate: provider configured AND params.req is not None AND
               exact < total AND the miss suffix is at least
               fuzzy_min_suffix_tokens. Internal re-matches
               (cache_unfinished_req) pass req=None and stay exact-only.
            3. provider.match_on_prefix_miss(...) -> FuzzyMatchResult | None
               (CDC chunks of the unmatched tail; mandatory token-ID
               equality check per chunk — never trust the hash alone)
            4. Validate: donor_last_node_id resolvable in _node_registry
               (stale donors from resets/evictions are dropped);
               len(kv_cache_indices) == cached_token_count.
            5. ALLOC (owner: recipient request): if the donor span is not
               position-aligned, pre-allocate cached_token_count slots ->
               req.fuzzy_realized_locs. Allocation failure = clean
               exact-only fallback; no request state was mutated.
            6. LOCK (owner: FuzzyRadixCache): inc_lock_ref(donor node) ->
               req.fuzzy_donor_node. Pinned donors cannot be LRU-evicted
               while any recipient is in flight.
            7. Return MatchResult with device_indices = exact ++ donor
               indices, fuzzy_matched_len, and cache_protected_len =
               exact+fuzzy (the freeing floor for cache_*_req).
  └─ req.cache_fuzzy_matched_len = fuzzy_matched_len

ForwardBatch.init_new                              (model_executor/forward_batch_info.py)
  └─ fuzzy_reqs = [reqs with cache_fuzzy_matched_len > 0]   (extend only)

ModelRunner._forward_raw                           (model_executor/model_runner.py)
  └─ FuzzyKVRealizer.realize(fuzzy_reqs)           (fuzzy_match/realizer.py)
       Runs on the forward stream after the decode-CUDA-graph early
       return and before the extend dispatch — never inside graph
       capture/replay (same placement as the deferred mamba COW hook).
       ROPE (owner: FuzzyKVRealizer): k_nope (position-free) copied;
       k_rope = apply_rope(new_pos, reverse_rope(donor_pos, k_rope_donor)).
       Layers flagged by layer_recompute_mask are zeroed instead (drop,
       not recompute). req_to_token[fuzzy span] repointed to the realized
       slots. Per-request state cleared in a finally block so
       chunked-prefill re-entry, decode, and retract-resume never
       re-trigger.

decode rounds                                       (unchanged)

FuzzyRadixCache.cache_finished_req                 (fuzzy_radix_cache.py)
  1. FREE (owner: recipient): reclaim req.fuzzy_realized_locs if the
     forward pass never consumed them (e.g. aborted request).
  2. RadixCache.cache_finished_req (super):
       insert(): realized slots are adopted by the recipient's tree path.
       _on_finished_insert hook (between insert and duplicate-freeing,
       while the request's slots are still live):
         provider.cache_on_request_finished(...)  -> register as donor,
             twice: chunked from SINK_TOKENS and from the request's own
             divergence point (divergence-aligned registration)
         provider.on_donor_inserted(node.id)      -> donor addressable by
                                                      TreeNode id
       free duplicates above req.cache_protected_len; dec_lock_ref(last_node).
  3. UNLOCK: dec_lock_ref(req.fuzzy_donor_node).
```

`_node_registry` (TreeNode.id -> node) is maintained by `FuzzyRadixCache`
overrides of `_insert_helper` / `_split_node` / `_delete_leaf` / `reset`,
so a donor reference is always either resolvable-and-pinnable or
detectably stale — never dangling.

## Configuration

| Flag | Default | Why it exists |
|---|---|---|
| `--radix-cache-backend fuzzy_match` | off | The enable switch; registers nothing and costs nothing when unset. |
| `--fuzzy-match-provider` | `ExactHash` | Provider selection; the interface admits out-of-tree providers. |
| `--fuzzy-min-match-length` | `16` | Skips fuzzy lookup behind weak partial exact anchors. |

Provider-internal tuning (CDC chunk geometry, and the minimum-suffix lookup
gate `fuzzy_min_suffix_tokens=256` that bounds no-hit overhead) lives in
`FuzzyMatchConfig` defaults, not CLI flags.

## Scope and guarantees

- Exact prefix matching is untouched and always wins; the fuzzy match runs
  only on the missed suffix. The default backend path is unchanged when the
  backend is not selected (three seams: one `MatchResult` field, one no-op
  hook in `cache_finished_req`, five inert `Req` fields).
- The mechanism is lossless *by construction* for the matching side (exact
  token-ID equality per chunk) and position-correct *by construction* for
  the KV side (reverse-RoPE at donor positions, forward-RoPE at recipient
  positions on the rope slice; the position-free latent slice copied
  verbatim). Measured divergence vs fresh recompute: avg KL ≈ 0.0032 on
  DeepSeek-V2-Lite-Chat / 0.0019 on Moonlight-16B-A3B-Instruct.
- Supported KV pools: `MLATokenToKVPool` with bf16 storage and
  `rotary_dim == qk_rope_head_dim` (DeepSeek-V2/V3-style MLA); the MHA path
  exists but is reference material for the MLA work.
- Not yet supported: FP8 KV cache (a known rotation-precision bug: cos/sin
  are cast to the 8-bit storage dtype by the rotary helpers), EAGLE
  speculative decoding, hybrid SSM/Mamba models, multi-region
  (`|exact|miss|fuzzy|miss|...`) reuse, hierarchical (host) cache
  interaction, TP > 1. Each unsupported case is rejected explicitly rather
  than silently.
