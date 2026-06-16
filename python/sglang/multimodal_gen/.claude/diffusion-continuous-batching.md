# Diffusion continuous batching (multimodal_gen)

Step-level continuous batching for the diffusion denoising loop. Opt-in, single-device,
MONOLITHIC role only. Default OFF.

Enable:
```
sglang serve ... --enable-diffusion-continuous-batching \
    [--diffusion-cb-max-running N] [--diffusion-cb-fuse-forward true|false] \
    [--diffusion-cb-steps-per-tick K]
```

## What it does
Inverts the synchronous scheduler control flow so several request *groups* denoise
concurrently and advance one step per "tick": groups admitted mid-flight join at
the next step, and finish independently (a 20-step group returns while a 50-step
one keeps going). The server stays responsive between steps.

## Architecture / files
- `runtime/managers/diffusion_continuous_batching.py` — `ContinuousBatchingEngine`,
  `DiffusionRequestState`, `StepRunner` protocol. Worker-agnostic + unit-tested
  (8 engine tests, no GPU needed).
- `runtime/pipelines_core/stages/denoising.py` — `cb_begin` / `cb_num_steps` /
  `cb_run_step` / `cb_end` / `cb_supports`, plus the fused path
  (`cb_run_step_fused`, `cb_group_fusable`, `_cb_merge_value`). Thin wrappers reuse
  the SAME per-step body as `forward()` (`_prepare_step_state` →
  `_run_denoising_step` → `_record_trajectory`), so behavior matches the
  monolithic path; the fused/per-request paths share `_build_forward_inputs` +
  `_apply_scheduler_step` and the `_split_batched_prediction` postprocess.
- `runtime/pipelines_core/composed_pipeline_base.py` — `cb_split_stages()`
  (pre / denoise / post), `cb_setup_residency()`, `cb_run_stages()`.
- `runtime/managers/gpu_worker.py` — implements `StepRunner` (`cb_prepare`,
  `cb_step`, `cb_finalize`, `cb_make_error_output`, `cb_can_run`). `cb_finalize`
  reuses `_execute_forward_common` for metrics/output-transport.
- `runtime/managers/scheduler.py` — `_cb_event_loop_iteration` (admit → dispatch
  ineligible → step), gated by `_maybe_init_cb_engine`.
- `runtime/server_args.py` — `enable_diffusion_continuous_batching`,
  `diffusion_cb_max_running`, `diffusion_cb_fuse_forward`,
  `diffusion_cb_steps_per_tick`.
- `test/test_diffusion_continuous_batching.py` — engine scheduling tests (no GPU).
- `test/test_diffusion_cb_fused.py` — proves fused == per-request latents on CPU
  with a fake transformer.

## Merged group admission (the throughput foundation)
Each running unit is a **merged batch group**, not a single request. The scheduler
collects compatible eligible requests (`_cb_collect_group`), merges them into one
batched `Req` (`_try_merge_generation_reqs`), and admits the group with its list of
client `identities` + original `Req`s. The worker runs the **batched** pre-denoise
stages (one T5 encode for N prompts), steps the batched denoise loop, and runs the
**batched** decode; the scheduler splits the batched OutputBatch back per client
(`_split_batched_output`). Across groups, the engine fuses compatible groups'
denoise forwards (`batch_key` keys on *per-sample* shape) and completes groups
independently.

Params: group/merge size = `diffusion_cb_max_running`; engine
`max_running = max(32, 4×group_size)`; fused-forward cap = `2×group_size`.

**This batched encode/denoise/decode is what makes CB throughput-competitive with
dynamic batching.** Without it (per-request encode/decode) CB ran at ~40% of
dynamic batching; with it, ~95%.

## Fused forward
`GPUWorker.cb_step` tries `DenoisingStage.cb_run_step_fused` first: it packs a
compatible group's scaled latents into ONE batched model forward (per-sample
timestep tensor, so requests at *different* steps share the pass), splits the
prediction, and advances each request's own scheduler. Toggle with
`--diffusion-cb-fuse-forward` (default on).

Guarded by `cb_group_fusable`: single transformer, dense attention (no
STA/VSA/SVG2), cache-dit off, no TeaCache, no Wan-TI2V, no active CFG gate, and
cond-kwargs that merge cleanly. Any miss → `cb_run_step_fused` returns `False`
(nothing mutated) → per-request fallback.

`cb_run_step_fused` branches on `do_classifier_free_guidance`:
- non-CFG: one fused forward over N requests (single conditional branch).
- CFG: builds a merged `CFGPolicy` (`_cb_build_merged_cfg_policy`) whose branch
  kwargs are concatenated across requests, then calls `_predict_noise_with_cfg`
  once over the N-request batch — it runs both branches (one 2N forward with
  `--enable-batched-cfg`, else two N forwards) and combines, returning N
  already-sliced per-request estimates.

The merge (`_cb_merge_value`) classifies each kwarg: per-sample tensors (leading
dim == batch size) are concatenated; shared tensors (rotary) pass through;
**nested** sequences (rotary `tuple[tuple[cos,sin],...]`) recurse; and a *list*
whose length == batch size is a **per-image** list (e.g. Z-Image's variable-length
captions) and is *extended* across requests. `batch_key` includes
`guidance_scale`/`true_cfg_scale`/`guidance_rescale` so the CFG combine's
batch-level postprocess (run on one representative request) is valid group-wide.

The merged conditioning (prompt embeds / masks / rotary / merged CFG policy /
guidance) is invariant across a group's denoise steps, so `cb_run_step_fused`
caches it keyed by the group's request ids rather than re-concatenating each step.

## Performance

### CB vs dynamic batching — FLUX.1-schnell 256×256, 4-step
1×H100 exclusive, resident weights (`--dit-cpu-offload false`), conc=16, 48 reqs:

| config | req/s | p50 latency | vs dynamic |
|---|---|---|---|
| Baseline (no batching) | 2.49 | 6.41s | 27% |
| **Dynamic batching** (`--batching-max-size 8 --batching-delay-ms 50`) | **9.08** | 1.74s | 100% |
| **CB, fused** | **8.68** | 1.82s | **96%** |
| CB, no fusion (`--diffusion-cb-fuse-forward false`) | 8.62 | 1.82s | 95% |

Takeaways:
- **Merged-group batching is the dominant win**: baseline → CB is ~3.5× (2.49 →
  8.6 req/s), entirely from batching the encode/denoise/decode like dynamic does.
- **At this workload fusion adds almost nothing** (8.62 → 8.68): a 12B forward at
  256×256 is not the bottleneck — the encode/decode batching is, and no-fusion
  already gets that. Fusion's value shows up elsewhere (below).
- CB reaches **96% of dynamic batching** while keeping mid-flight admission,
  independent completion, and cross-group fusion. The ~4% gap is per-step Python
  orchestration (CB runs a step at a time; dynamic runs the whole loop per call).
- CB also gives **tighter latency** (p50 ≈ p99) regardless.

### When fusion helps — single-forward GPU-utilization headroom
Fusion only speeds things up when one denoise forward *under-utilizes* the GPU;
once a single request saturates it, fusion gives nothing. Per-model conc=16 vs
conc=1 (image models, 1×H100 exclusive, resident weights):

| Model / res | single-stream util | fusion speedup |
|---|---|---|
| Z-Image 256 | 51% | **1.58×** |
| FLUX 256 | 76% | 1.28× |
| FLUX 512 | 88% | 1.07× |
| Z-Image 512 | 83% | ~0.95× (slight regress) |
| Z-Image 768 | 96% | ~0.93× |
| FLUX 1024 | 95% | 1.00× |

It can slightly regress when there's little headroom *and* many steps (Z-Image
512/768, 8-step): CB's per-step Python overhead outweighs the small fusion gain.
Reducing per-step CB overhead is the remaining lever here.

### Open-loop / streaming — adaptive admission gate
Under Poisson arrivals, eager admission every tick makes 1–2 request groups, so
the (expensive) text-encode runs nearly per-request and lands on the per-step
critical path, blocking every in-flight request's denoise (~2× dynamic's latency).

**Fix — `_cb_should_admit`:** don't drain the queue every tick; let it accumulate
into a fuller group before encoding. Admit when (a) the engine is **idle** (don't
waste the GPU — low latency at low load), (b) a **full group** is queued, or (c)
the queue head has **waited** `_cb_admission_delay_s` (default 100ms,
`--batching-delay-ms` overrides). The wait overlaps with running groups' denoise
steps, so it is not idle latency.

**Step amortization** (`--diffusion-cb-steps-per-tick`, default 3): take several
denoise steps per event-loop iteration to amortize the per-step recv/admit
round-trip, breaking early to admit a ready group or when the engine drains. Plus
per-request eligibility caching (the admission scan ran `cb_can_run` per queued
req per tick). This raises CB's throughput ceiling near saturation.

Result (FLUX-256/4-step, p50 e2e latency at matched throughput):

| arrivals | dynamic (50ms) | CB |
|---|---|---|
| 2/s | 0.48s | 0.66s |
| 6/s | 0.85s | **0.69s** (CB wins) |
| 8/s | 0.94s | **0.88s** (CB matches/wins) |
| 10/s | — | 1.33s (past the ~7.7/s ceiling) |

CB matches or beats dynamic batching from low rate through saturation — it admits
immediately when idle, where dynamic always pays its fixed delay window. What did
NOT help: capping concurrent groups, and longer admission delays — both reduce
in-flight parallelism and starve the GPU.

### Mixed step counts — CB's killer app, gated on step-count DIVERSITY
Dynamic batching can only merge requests with **identical** `num_inference_steps`.
CB fuses across *different-progress* requests. The crossover depends on how diverse
the step counts are:

- **Clustered** (steps spread over ~7 distinct values, e.g. {8,12,16,20,28,36,50}):
  dynamic still forms full batches of 8 and stays ahead. FLUX-256, conc=16,
  total=56, 1×H100 exclusive:

  | | req/s | SHORT(≤16) p50 | LONG(≥36) p50 |
  |---|---|---|---|
  | Dynamic batching | **1.43** | 6.68s | **10.79s** |
  | Continuous batching | 1.27 | 6.09s | 20.32s |

  CB is ~12% slower and its long requests wait ~2× longer (they sit through every
  group's per-tick denoise). With only 7 distinct step counts dynamic keeps full
  batches, so it stays ahead.

- **Near-unique** (every request a distinct step count): dynamic degenerates to
  **batch size 1, fully sequential**; CB fuses everything. FLUX-256, distinct steps
  10..41, conc = total = 32 (all in flight), 1×H100 exclusive:

  | | req/s | SHORT(≤16) p50 | LONG(≥36) p50 |
  |---|---|---|---|
  | Dynamic batching | 0.68 | **3.17s** | 41.84s |
  | **Continuous batching** | **1.77** | 9.29s | **17.84s** |

  CB is **2.6× the throughput** and also beats dynamic on long-request latency
  (17.84s vs 41.84s). No tuning needed — cross-group fusion already batches
  everything. Tradeoff: **worse short-request latency** (short reqs wait inside the
  big fused batch: 9.29s vs dynamic's 3.17s), so CB trades tail latency for
  throughput.

**Recommendation:** enable CB for diverse-step workloads (and to overlap admission/
completion); keep dynamic batching as the default for uniform/clustered ones, where
CB's per-step orchestration makes it a peer at best.

### Wan video (Wan2.1-T2V-1.3B, 1×H100) — CFG fusion works
GPU-validated: `fused 3/4 requests into one denoise step (CFG)`, 4 distinctive
prompts → 4 correctly-aligned videos. Load test (320×576, 17 frames, submit-conc 4,
8 videos): 8-step 0.615 → **0.668** vid/s (1.09×), 20-step 0.415 → **0.426**
(1.03×). Modest — a Wan video forward already nearly saturates the H100, so
fusion has little compute headroom; correctness (aligned videos) is the win.
NOTE: the CB path's reported `inference_time_s` is unreliable (it times only
finalize); use wall-clock / e2e latency.

## Still falls back to the synchronous path
rollout, sequence parallelism, warmup, multi-GPU, disagg, sparse attention,
cache-dit, TeaCache, and any model whose conditioning kwargs don't merge (safe
fallback, logged once).

**Custom-loop denoising stages** (`cb_supports`): a `DenoisingStage` subclass that
overrides `forward` has a custom loop the cb_* hooks (base prepare/step/finalize)
don't reproduce — e.g. **LTX-2's joint audio+video** `LTX2AVDenoisingStage`, the
DMD / causal-DMD / Hunyuan3D-shape / Sana-WM / Ideogram stages. Detected via
`type(self).forward is not DenoisingStage.forward` and fall back to the synchronous
path (otherwise CB would silently run the base video-only loop and drop the model's
extra work, e.g. audio). Subclasses that override only step *helpers*
(`_run_denoising_step`) keep the base loop and still fuse.

## Correctness notes (surfaced by GPU testing)
1. **Router unwrapping** (`composed_pipeline_base._cb_unwrap_denoising`): FLUX and
   most image models wrap the denoising stage in `ProgressiveDenoisingStageRouter`,
   which delegates `fullres` (default) requests to a plain `DenoisingStage`. CB
   unwraps the router and drives that inner stage; `cb_can_run` rejects actual
   progressive-resolution requests.
2. **Per-request scheduler isolation** (`GPUWorker.cb_prepare`): the timestep-prep
   stage hands every request the stage's *shared* scheduler runtime; concurrent
   stepping corrupts its `_step_index`/sigmas. Fixed by
   `clone_scheduler_runtime(req.scheduler)` per request before `cb_begin`.
3. **Fused-group cap** (`_maybe_init_cb_engine`): the engine `max_batch_size`
   derives from `diffusion_cb_max_running` (not `--batching-max-size`, whose
   default of 1 would cap every fused group at 1 so fusion never happened).
4. **Z-Image / Wan merge generalizations**: nested rotary tuples must recurse, and
   per-encoder context lists must be handled by structure — Z-Image's per-image
   caption list (`[tensor(L,D)]`, no batch dim) is *extended* across requests,
   while Wan's `[tensor(1,512,D)]` context (batch dim inside) is *recursed* and the
   inner tensor concatenated. Fused output for Z-Image is not bit-identical to
   single-request (variable-length captions pad to batch-max — inherent to batched
   Z-Image, shared by its native batching); images remain correct. FLUX and other
   size-invariant-conditioning models fuse faithfully.

## Async encode/decode overlap — investigated, not pursued
Overlapping the heavy text-encode / VAE-decode with denoise on a background thread
was prototyped and **fully reverted**: diffusion's optimized norm/attention kernels
(CuTe DSL + TVM-FFI, lazy per-shape compile/launch) are **not thread-safe** —
concurrent encode+denoise forwards corrupt each other, and the only correctness fix
(a GPU lock serializing forwards) removes the overlap entirely. The viable path is
to sidestep multi-thread launches: single-thread multi-CUDA-stream, CUDA MPS, or a
separate encode/decode process. Not pursued; stages stay inline (which preserves
the open-loop wins above). Nothing from the prototype was kept.
