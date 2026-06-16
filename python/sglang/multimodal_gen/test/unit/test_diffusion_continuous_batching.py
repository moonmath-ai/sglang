# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the diffusion continuous-batching framework.

Combines:
1. Engine scheduling tests (independent completion, admission, caps, etc.).
2. Denoising stage fused forward equivalence tests (Euler/EulerCFG simulation).

These run on CPU without requiring a GPU or any model weights.
"""

import types
import unittest

import torch

from sglang.multimodal_gen.runtime.managers.diffusion_continuous_batching import (
    ContinuousBatchingEngine,
    DiffusionRequestState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage

# =====================================================================
# 1. Engine Scheduling Test Helpers & Suites
# =====================================================================


class _FakeEngineReq:
    def __init__(self, request_id, num_steps, batch_key, prepare_error=None):
        self.request_id = request_id
        self.num_steps = num_steps
        self.batch_key = batch_key
        self.prepare_error = prepare_error


class _FakeRunner:
    """Records the order/grouping of steps and produces deterministic outputs."""

    def __init__(self, step_error_at=None):
        # step_error_at: dict[request_id] -> step_index that should raise.
        self.step_error_at = step_error_at or {}
        self.step_groups: list[list[str]] = []
        self.finalized: list[str] = []

    def cb_prepare(
        self, identities, original_reqs, merged_req
    ) -> DiffusionRequestState:
        state = DiffusionRequestState(
            identities=list(identities),
            original_reqs=list(original_reqs),
            request_id=merged_req.request_id,
            batch_key=merged_req.batch_key,
            num_steps=merged_req.num_steps,
            req=merged_req,
        )
        if merged_req.prepare_error is not None:
            state.error = merged_req.prepare_error
        return state

    def cb_step(self, group: list[DiffusionRequestState]) -> None:
        self.step_groups.append([s.request_id for s in group])
        for state in group:
            if self.step_error_at.get(state.request_id) == state.step_index:
                state.error = f"boom at step {state.step_index}"
                continue
            state.step_index += 1

    def cb_finalize(self, state: DiffusionRequestState):
        self.finalized.append(state.request_id)
        return {"request_id": state.request_id, "ok": True, "n": state.num_requests}

    def cb_make_error_output(self, state: DiffusionRequestState):
        return {"request_id": state.request_id, "error": state.error}


def _admit1(engine, identity, req):
    """Admit a single-request group."""
    return engine.admit([identity], [req], req)


def _drain(engine: ContinuousBatchingEngine):
    """Step until idle, returning finished (request_id, output) pairs in order."""
    done = []
    while engine.has_work():
        for _state, output in engine.step():
            done.append((output["request_id"], output))
    return done


class TestContinuousBatchingEngine(unittest.TestCase):
    def test_independent_completion_order(self):
        """A short group finishes while a longer one keeps running."""
        runner = _FakeRunner()
        engine = ContinuousBatchingEngine(runner, max_running=8, max_batch_size=8)

        self.assertIsNone(_admit1(engine, b"a", _FakeEngineReq("a", 2, "k")))
        self.assertIsNone(_admit1(engine, b"b", _FakeEngineReq("b", 5, "k")))

        self.assertEqual(engine.step(), [])  # tick 1
        finished_t2 = engine.step()  # "a" (2 steps) finishes
        self.assertEqual([o["request_id"] for _s, o in finished_t2], ["a"])
        self.assertEqual(engine.num_running, 1)

        rest = _drain(engine)
        self.assertEqual([rid for rid, _o in rest], ["b"])
        self.assertEqual(runner.finalized, ["a", "b"])

    def test_midflight_admission(self):
        """A group admitted after others have advanced still completes."""
        runner = _FakeRunner()
        engine = ContinuousBatchingEngine(runner, max_running=8, max_batch_size=8)

        _admit1(engine, b"a", _FakeEngineReq("a", 3, "k"))
        engine.step()  # a -> step 1
        _admit1(engine, b"late", _FakeEngineReq("late", 1, "k"))

        finished = _drain(engine)
        ids = [rid for rid, _o in finished]
        self.assertEqual(ids, ["late", "a"])

    def test_merged_group_carries_identities(self):
        """A group of 3 requests is one running state with 3 identities."""
        runner = _FakeRunner()
        engine = ContinuousBatchingEngine(runner, max_running=8, max_batch_size=8)
        reqs = [_FakeEngineReq(f"r{i}", 1, "k") for i in range(3)]
        merged = _FakeEngineReq("merged", 1, "k")
        self.assertIsNone(engine.admit([b"r0", b"r1", b"r2"], reqs, merged))
        self.assertEqual(engine.num_running, 3)  # 3 requests
        self.assertEqual(engine.num_running_groups, 1)  # in 1 group
        finished = engine.step()
        self.assertEqual(len(finished), 1)
        state, output = finished[0]
        self.assertEqual(state.identities, [b"r0", b"r1", b"r2"])
        self.assertEqual(output["n"], 3)

    def test_cross_group_fusion_capped_by_request_count(self):
        """max_batch_size caps the fused forward by total requests, not groups."""
        runner = _FakeRunner()
        engine = ContinuousBatchingEngine(runner, max_running=16, max_batch_size=5)
        # two groups of 3 (same key) -> 6 requests > cap 5 -> two fused sub-batches
        engine.admit(
            [b"a0", b"a1", b"a2"],
            [_FakeEngineReq(f"a{i}", 1, "k") for i in range(3)],
            _FakeEngineReq("A", 1, "k"),
        )
        engine.admit(
            [b"b0", b"b1", b"b2"],
            [_FakeEngineReq(f"b{i}", 1, "k") for i in range(3)],
            _FakeEngineReq("B", 1, "k"),
        )
        engine.step()
        # First sub-batch = [A] (3); adding B (3) would exceed 5 -> [B] separate.
        self.assertEqual(runner.step_groups, [["A"], ["B"]])

    def test_prepare_error_returns_immediately(self):
        runner = _FakeRunner()
        engine = ContinuousBatchingEngine(runner, max_running=8, max_batch_size=8)
        result = _admit1(engine, b"bad", _FakeEngineReq("bad", 3, "k", prepare_error="nope"))
        self.assertIsNotNone(result)
        _state, output = result
        self.assertEqual(output["error"], "nope")
        self.assertFalse(engine.has_work())

    def test_zero_step_request_finalized_on_admit(self):
        runner = _FakeRunner()
        engine = ContinuousBatchingEngine(runner, max_running=8, max_batch_size=8)
        result = _admit1(engine, b"z", _FakeEngineReq("z", 0, "k"))
        self.assertIsNotNone(result)
        _state, output = result
        self.assertTrue(output["ok"])
        self.assertEqual(runner.finalized, ["z"])

    def test_step_error_is_isolated(self):
        """One group failing mid-step does not stop its groupmate."""
        runner = _FakeRunner(step_error_at={"a": 1})
        engine = ContinuousBatchingEngine(runner, max_running=8, max_batch_size=8)
        _admit1(engine, b"a", _FakeEngineReq("a", 3, "k"))
        _admit1(engine, b"b", _FakeEngineReq("b", 3, "k"))

        outputs = {}
        while engine.has_work():
            for _s, o in engine.step():
                outputs[o["request_id"]] = o

        self.assertIn("error", outputs["a"])
        self.assertTrue(outputs["b"].get("ok"))

    def test_capacity_limits_admission(self):
        runner = _FakeRunner()
        engine = ContinuousBatchingEngine(runner, max_running=1, max_batch_size=8)
        _admit1(engine, b"a", _FakeEngineReq("a", 2, "k"))
        self.assertEqual(engine.admit_headroom(), 0)


# =====================================================================
# 2. Fused Forward Equivalence Test Helpers & Suites
# =====================================================================


class _FakeScheduler:
    """Euler-ish scheduler: x_{t+1} = x_t - 0.1 * noise_pred. Per-request state."""

    order = 1

    def scale_model_input(self, sample, t):
        return sample

    def step(self, *, model_output, timestep, sample, return_dict=False, **kw):
        return (sample - 0.1 * model_output,)


class _FakeTransformer(torch.nn.Module):
    """Per-sample independent map: depends on hidden_states, timestep, embeds.

    Each sample in the batch is transformed independently, and the shared
    positional tensor `pos` (no batch dim) is added broadcast — so a correct
    fused forward must NOT concatenate `pos`.
    """

    def forward(self, *, hidden_states, timestep, encoder_hidden_states, pos, **kw):
        t = timestep.to(hidden_states.dtype).unsqueeze(-1)
        return hidden_states * 0.5 + 0.01 * t + 0.1 * encoder_hidden_states + pos


def _make_stage() -> DenoisingStage:
    stage = DenoisingStage.__new__(DenoisingStage)
    stage.transformer = None
    stage.transformer_2 = None
    stage._cache_dit_enabled = False
    stage.attn_backend = None
    stage._extra_func_kwarg_names_cache = {}
    return stage


def _make_ctx(stage, latents, embeds, pos, transformer, neg_embeds=None):
    from sglang.multimodal_gen.runtime.distributed.cfg_policy import (
        CFGBranch,
        CFGPolicy,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
        DenoisingContext,
    )

    branches = [
        CFGBranch("conditional", True, {"encoder_hidden_states": embeds, "pos": pos})
    ]
    if neg_embeds is not None:
        branches.append(
            CFGBranch(
                "unconditional",
                False,
                {"encoder_hidden_states": neg_embeds, "pos": pos},
            )
        )
    policy = CFGPolicy(branches=branches)
    ts = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0])
    ctx = DenoisingContext(
        scheduler=_FakeScheduler(),
        extra_step_kwargs={},
        target_dtype=torch.float32,
        autocast_enabled=False,
        timesteps=ts,
        num_inference_steps=5,
        num_warmup_steps=0,
        image_kwargs={},
        pos_cond_kwargs={},
        neg_cond_kwargs={},
        latents=latents,
        boundary_timestep=None,
        z=None,
        reserved_frames_mask=None,
        seq_len=None,
        guidance=None,
        is_warmup=False,
        cfg_policy=policy,
    )
    ctx.extra["cb_timesteps_cpu"] = ts
    ctx.extra["cb_num_timesteps"] = ts.shape[0]
    ctx.extra["cb_use_nvtx"] = False
    return ctx


class _FakeFusedReq:
    def __init__(self, do_cfg=False):
        self.do_classifier_free_guidance = do_cfg
        self.enable_teacache = False
        self.image_latent = None
        self.is_cfg_negative = False
        self.raw_latent_shape = (1, 4)
        self.cfg_normalization = 0.0
        self.guidance_rescale = 0.0
        self.guidance_scale = 3.0

    def __getattr__(self, name):
        return None


CFG_SCALE = 3.0
batch_bsz = 1


def _fake_server_args(transformer):
    pipeline_config = types.SimpleNamespace(
        slice_noise_pred=lambda pred, latents: pred,
        task_type=None,
        get_classifier_free_guidance_scale=lambda batch, gs: CFG_SCALE,
        postprocess_cfg_noise=lambda batch, noise_pred, noise_pred_cond: noise_pred,
    )
    return types.SimpleNamespace(
        comfyui_mode=False,
        enable_cfg_parallel=False,
        pipeline_config=pipeline_config,
    )


class TestFusedStepEquivalence(unittest.TestCase):
    def _patch_stage(self, stage, transformer):
        stage._prepare_step_state = lambda ctx, batch, sa, i, t_host, ts_cpu: (
            types.SimpleNamespace(
                step_index=i,
                t_host=t_host,
                t_device=ts_cpu[i],
                t_int=int(t_host.item()),
                current_model=transformer,
                current_guidance_scale=None,
                attn_metadata=None,
            )
        )
        stage.expand_timestep_before_forward = (
            lambda batch, sa, t_device, dtype, seq_len, mask: t_device.repeat(batch_bsz)
        )
        stage._record_trajectory = lambda *a, **k: None
        stage.step_profile = lambda: None
        stage.post_forward_for_ti2v_task = lambda b, sa, m, latents, z: latents
        stage._predict_noise = lambda current_model, latent_model_input, timestep, target_dtype, guidance, **kw: current_model(
            hidden_states=latent_model_input, timestep=timestep, **kw
        )

    def test_fused_equals_sequential(self):
        global batch_bsz
        batch_bsz = 1
        torch.manual_seed(0)
        transformer = _FakeTransformer()
        sa = _fake_server_args(transformer)

        # Two requests at DIFFERENT steps (different progress), same shapes.
        latents = [torch.randn(1, 4), torch.randn(1, 4)]
        embeds = [torch.randn(1, 4), torch.randn(1, 4)]
        pos = torch.randn(4)  # shared positional, no batch dim -> pass-through
        step_indices = [3, 1]

        # --- ground truth: each request stepped alone, computed directly ---
        ts = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0])
        ref_latents = []
        for i in range(2):
            t = ts[step_indices[i]]
            noise = transformer(
                hidden_states=latents[i],
                timestep=t.repeat(1),
                encoder_hidden_states=embeds[i],
                pos=pos,
            )
            ref_latents.append(latents[i] - 0.1 * noise)

        # --- fused: both in one forward ---
        stage = _make_stage()
        self._patch_stage(stage, transformer)
        ctxs = [
            _make_ctx(stage, latents[i].clone(), embeds[i], pos, transformer)
            for i in range(2)
        ]
        reqs = [_FakeFusedReq(), _FakeFusedReq()]
        ok = stage.cb_run_step_fused(ctxs, reqs, step_indices, sa)
        self.assertTrue(ok, "expected the group to fuse")

        for i in range(2):
            torch.testing.assert_close(ctxs[i].latents, ref_latents[i])

    def test_cfg_fused_equals_sequential(self):
        """CFG: a fused N-request step matches per-request cond/uncond + combine."""
        global batch_bsz
        batch_bsz = 1
        torch.manual_seed(1)
        transformer = _FakeTransformer()
        sa = _fake_server_args(transformer)

        latents = [torch.randn(1, 4), torch.randn(1, 4)]
        cond = [torch.randn(1, 4), torch.randn(1, 4)]
        uncond = [torch.randn(1, 4), torch.randn(1, 4)]
        pos = torch.randn(4)
        step_indices = [3, 1]
        ts = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0])

        # --- ground truth: per-request CFG = uncond + scale*(cond - uncond) ---
        ref_latents = []
        for i in range(2):
            t = ts[step_indices[i]].repeat(1)
            p = transformer(
                hidden_states=latents[i],
                timestep=t,
                encoder_hidden_states=cond[i],
                pos=pos,
            )
            n = transformer(
                hidden_states=latents[i],
                timestep=t,
                encoder_hidden_states=uncond[i],
                pos=pos,
            )
            combined = n + CFG_SCALE * (p - n)
            ref_latents.append(latents[i] - 0.1 * combined)

        # --- fused CFG: both requests, both branches, advanced together ---
        stage = _make_stage()
        self._patch_stage(stage, transformer)
        ctxs = [
            _make_ctx(
                stage,
                latents[i].clone(),
                cond[i],
                pos,
                transformer,
                neg_embeds=uncond[i],
            )
            for i in range(2)
        ]
        reqs = [_FakeFusedReq(do_cfg=True), _FakeFusedReq(do_cfg=True)]
        ok = stage.cb_run_step_fused(ctxs, reqs, step_indices, sa)
        self.assertTrue(ok, "expected the CFG group to fuse")

        for i in range(2):
            torch.testing.assert_close(ctxs[i].latents, ref_latents[i])

    def test_merge_passes_through_shared_tensor(self):
        stage = _make_stage()
        pos = torch.randn(7, 4)  # leading dim != base_bs -> shared
        merged = stage._cb_merge_value([pos, pos], base_bs=1)
        self.assertIs(merged, pos)

        embeds = [torch.randn(1, 3, 4), torch.randn(1, 3, 4)]
        cat = stage._cb_merge_value(embeds, base_bs=1)
        self.assertEqual(tuple(cat.shape), (2, 3, 4))

    def test_merge_rejects_mismatched(self):
        stage = _make_stage()
        bad = [torch.randn(1, 3, 4), torch.randn(1, 5, 4)]  # shape[1:] differ
        self.assertIs(stage._cb_merge_value(bad, base_bs=1), False)

    def test_fused_conditioning_cached_across_steps(self):
        """The invariant conditioning merge runs once per group, not per step."""
        global batch_bsz
        batch_bsz = 1
        torch.manual_seed(2)
        transformer = _FakeTransformer()
        sa = _fake_server_args(transformer)
        latents = [torch.randn(1, 4), torch.randn(1, 4)]
        embeds = [torch.randn(1, 4), torch.randn(1, 4)]
        pos = torch.randn(4)

        stage = _make_stage()
        self._patch_stage(stage, transformer)
        calls = {"n": 0}
        real_merge = stage._cb_merge_request_kwargs

        def counting_merge(kwargs_list, base_bs):
            calls["n"] += 1
            return real_merge(kwargs_list, base_bs)

        stage._cb_merge_request_kwargs = counting_merge

        ctxs = [
            _make_ctx(stage, latents[i].clone(), embeds[i], pos, transformer)
            for i in range(2)
        ]
        reqs = [_FakeFusedReq(), _FakeFusedReq()]
        for s in range(3):
            ok = stage.cb_run_step_fused(ctxs, reqs, [s, s], sa)
            self.assertTrue(ok)
        self.assertEqual(calls["n"], 1, "conditioning should merge once, then cache")

    def test_merge_per_image_caption_list(self):
        """A per-image caption list (len == base_bs) is extended across requests."""
        stage = _make_stage()
        cap0 = [torch.randn(7, 8)]
        cap1 = [torch.randn(11, 8)]
        merged = stage._cb_merge_value([cap0, cap1], base_bs=1)
        self.assertEqual(len(merged), 2)
        self.assertIs(merged[0], cap0[0])
        self.assertIs(merged[1], cap1[0])

    def test_merge_per_encoder_list_with_batch_dim(self):
        """A length-1 list of batch-dim tensors (Wan-style context) concatenates
        the inner tensor and keeps list length 1 (not extended)."""
        stage = _make_stage()
        ctx0 = [torch.randn(1, 512, 16)]
        ctx1 = [torch.randn(1, 512, 16)]
        merged = stage._cb_merge_value([ctx0, ctx1], base_bs=1)
        self.assertEqual(len(merged), 1)
        self.assertEqual(tuple(merged[0].shape), (2, 512, 16))

    def test_merge_tuple_not_extended_when_len_equals_base_bs(self):
        """A (cos, sin) tuple must recurse, not extend, even if len == base_bs."""
        stage = _make_stage()
        cos = torch.randn(1024, 64)
        sin = torch.randn(1024, 64)
        freqs = (cos, sin)
        merged = stage._cb_merge_value([freqs, freqs], base_bs=2)
        self.assertIsInstance(merged, tuple)
        self.assertEqual(len(merged), 2)
        self.assertIs(merged[0], cos)

    def test_merge_nested_tuple_rotary(self):
        """Nested rotary-style tuple[tuple[cos,sin],...] of shared tensors fuses."""
        stage = _make_stage()
        text = (torch.randn(32, 64), torch.randn(32, 64))
        img = (torch.randn(4096, 64), torch.randn(4096, 64))
        freqs = (text, img)
        merged = stage._cb_merge_value([freqs, freqs, freqs], base_bs=1)
        self.assertIsNot(merged, False)
        self.assertEqual(len(merged), 2)
        self.assertIs(merged[0][0], text[0])
        self.assertIs(merged[1][1], img[1])


if __name__ == "__main__":
    unittest.main()
