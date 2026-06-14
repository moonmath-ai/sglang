# SPDX-License-Identifier: Apache-2.0
"""LTX-2 diffusers baseline vs native LiteLinear (no SGLang)."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import LTX2Pipeline
from lite_linear import LiteLinear

DEFAULT_MODEL = Path(
    "/mnt/fs/huggingface_cache/hub/models--Lightricks--LTX-2/snapshots/"
    "47da56e2ad66ce4125a9922b4a8826bf407f9d0a"
)
DEFAULT_DISTILLED_TRANSFORMER = DEFAULT_MODEL / "ltx-2-19b-distilled.safetensors"
DEFAULT_PROMPT = (
    "A quiet coastal town at sunrise, fishing boats moving through "
    "golden mist, cinematic camera movement"
)
DEFAULT_NEG = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted "
    "proportions, unnatural skin tones, deformed facial features, asymmetrical face, "
    "missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts "
    "around text, inconsistent perspective, camera shake, incorrect depth of field, "
    "background too sharp, background clutter, distracting reflections, harsh shadows, "
    "inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, "
    "unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, "
    "exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted "
    "audio, distorted voice, robotic voice, echo, background noise, off-sync audio, "
    "incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, "
    "flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI "
    "artifacts."
)

WORKLOAD_PRESETS: dict[str, dict[str, int | str | float | None]] = {
    "small": {
        "height": 512,
        "width": 768,
        "num_frames": 25,
        "num_inference_steps": 30,
        "guidance_scale": 4.0,
        "transformer_weights_path": None,
    },
    "moonmath": {
        "height": 704,
        "width": 1216,
        "num_frames": 72,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "transformer_weights_path": str(DEFAULT_DISTILLED_TRANSFORMER),
    },
}


@dataclass
class RunStats:
    mode: str
    load_s: float
    materialize_s: float
    total_s: float
    denoise_s: float
    transformer_forward_s: float
    step_ms: list[float]
    transformer_step_ms: list[float]
    peak_reserved_mb: float | None


def parse_args():
    p = argparse.ArgumentParser(description="LTX-2 diffusers LiteLinear on/off benchmark")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL))
    p.add_argument("--mode", choices=("default", "litelinear"), required=True)
    p.add_argument(
        "--workload",
        choices=tuple(WORKLOAD_PRESETS.keys()),
        default="small",
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative-prompt", default=DEFAULT_NEG)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument(
        "--transformer-weights-path",
        default=None,
        help="Optional single-file transformer checkpoint (e.g. distilled safetensors).",
    )
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--result-dir", default="/tmp/ltx_diffusers_litelinear")
    p.add_argument("--cpu-offload", action="store_true", default=False)
    p.add_argument(
        "--vae-cpu",
        action="store_true",
        default=False,
        help="Keep VAE on CPU (needs accelerate for diffusers offload hooks).",
    )
    return p.parse_args()


def resolve_workload(args) -> None:
    preset = WORKLOAD_PRESETS[args.workload]
    if args.height is None:
        args.height = int(preset["height"])
    if args.width is None:
        args.width = int(preset["width"])
    if args.num_frames is None:
        args.num_frames = int(preset["num_frames"])
    if args.num_inference_steps is None:
        args.num_inference_steps = int(preset["num_inference_steps"])
    if args.transformer_weights_path is None and preset.get("transformer_weights_path"):
        args.transformer_weights_path = str(preset["transformer_weights_path"])
    if args.guidance_scale is None:
        args.guidance_scale = float(preset.get("guidance_scale", 4.0))


def patch_transformer_ffn(transformer, rank: int) -> int:
    count = 0
    for block in transformer.transformer_blocks:
        for name in ("ff", "audio_ff"):
            ff = getattr(block, name)
            LiteLinear.replace_activation_proj_(ff.net[0])
            old = ff.net[2]
            lite = LiteLinear(
                old.in_features,
                old.out_features,
                bias=old.bias is not None,
                rank=rank,
                device=old.weight.device,
                dtype=old.weight.dtype,
            )
            with torch.no_grad():
                lite.weight.copy_(old.weight)
                if old.bias is not None:
                    lite.bias.copy_(old.bias)
            ff.net[2] = lite
            count += 2
    return count


def materialize_litelinear_modules(module: torch.nn.Module) -> int:
    n = 0
    for m in module.modules():
        if isinstance(m, LiteLinear) and getattr(m, "weight", None) is not None:
            m.materialize_from_weight()
            n += 1
    return n


def peak_reserved_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_reserved() / (1024**2)


def _place_pipeline(pipe: LTX2Pipeline, cpu_offload: bool, vae_cpu: bool) -> None:
    if cpu_offload:
        pipe.enable_model_cpu_offload()
        return
    pipe.to("cuda")
    if vae_cpu:
        for name in ("vae", "audio_vae", "vocoder"):
            mod = getattr(pipe, name, None)
            if mod is not None:
                mod.to("cpu")


def _load_transformer_weights(pipe: LTX2Pipeline, weights_path: str) -> None:
    from safetensors.torch import load_file

    state_dict = load_file(weights_path)
    pipe.transformer.load_state_dict(state_dict, strict=False)


def _install_transformer_forward_timer(
    transformer: torch.nn.Module,
) -> tuple[list[float], object]:
    step_ms: list[float] = []
    step_t0: list[float] = []

    def pre_hook(_module, _inputs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_t0[:] = [time.perf_counter()]

    def post_hook(_module, _inputs, _outputs):
        if not step_t0:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - step_t0[0]) * 1000.0)
        step_t0.clear()

    handles = [
        transformer.register_forward_pre_hook(pre_hook),
        transformer.register_forward_hook(post_hook),
    ]
    return step_ms, handles


def run_mode(args) -> RunStats:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    pipe = LTX2Pipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    if args.transformer_weights_path:
        _load_transformer_weights(pipe, args.transformer_weights_path)
    _place_pipeline(pipe, args.cpu_offload, args.vae_cpu)
    load_s = time.perf_counter() - t0

    materialize_s = 0.0
    if args.mode == "litelinear":
        t1 = time.perf_counter()
        patched = patch_transformer_ffn(pipe.transformer, rank=args.rank)
        pipe.transformer.eval()
        done = materialize_litelinear_modules(pipe.transformer)
        if patched != done:
            raise RuntimeError(
                f"LiteLinear patch count mismatch: patched={patched}, materialized={done}"
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        materialize_s = time.perf_counter() - t1

    step_ms: list[float] = []
    step_t0: list[float] = []
    transformer_step_ms, transformer_handles = _install_transformer_forward_timer(
        pipe.transformer
    )

    def on_step_begin(pipe_obj, step, timestep, callback_kwargs):  # noqa: ARG001
        step_t0[:] = [time.perf_counter()]
        return callback_kwargs

    def on_step_end(pipe_obj, step, timestep, callback_kwargs):  # noqa: ARG001
        now = time.perf_counter()
        if step_t0:
            step_ms.append((now - step_t0[0]) * 1000.0)
        return callback_kwargs

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_infer = time.perf_counter()
    try:
        _ = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
            output_type="pil",
            callback_on_step_begin=on_step_begin,
            callback_on_step_end=on_step_end,
        )
    finally:
        for handle in transformer_handles:
            handle.remove()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_s = time.perf_counter() - t_infer
    denoise_s = sum(step_ms) / 1000.0
    transformer_forward_s = sum(transformer_step_ms) / 1000.0

    return RunStats(
        mode=args.mode,
        load_s=load_s,
        materialize_s=materialize_s,
        total_s=total_s,
        denoise_s=denoise_s,
        transformer_forward_s=transformer_forward_s,
        step_ms=step_ms,
        transformer_step_ms=transformer_step_ms,
        peak_reserved_mb=peak_reserved_mb(),
    )


def main():
    args = parse_args()
    resolve_workload(args)
    out_dir = Path(args.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = run_mode(args)
    payload = {
        "mode": stats.mode,
        "load_s": stats.load_s,
        "materialize_s": stats.materialize_s,
        "inference_total_s": stats.total_s,
        "denoise_sum_s": stats.denoise_s,
        "transformer_forward_sum_s": stats.transformer_forward_s,
        "denoise_steps_ms": [
            {"step": i, "duration_ms": ms} for i, ms in enumerate(stats.step_ms)
        ],
        "transformer_forward_steps_ms": [
            {"step": i, "duration_ms": ms}
            for i, ms in enumerate(stats.transformer_step_ms)
        ],
        "peak_reserved_mb": stats.peak_reserved_mb,
        "workload": {
            "preset": args.workload,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
            "guidance_scale": args.guidance_scale,
            "transformer_weights_path": args.transformer_weights_path,
        },
    }
    out_path = out_dir / f"{stats.mode}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    s0 = stats.step_ms[0] if stats.step_ms else 0.0
    s_rest = sum(stats.step_ms[1:]) / max(len(stats.step_ms) - 1, 1)
    tf0 = stats.transformer_step_ms[0] if stats.transformer_step_ms else 0.0
    tf_rest = sum(stats.transformer_step_ms[1:]) / max(
        len(stats.transformer_step_ms) - 1, 1
    )
    print(f"mode={stats.mode} workload={args.workload}")
    print(f"  load={stats.load_s:.1f}s materialize={stats.materialize_s:.1f}s")
    print(f"  inference_total={stats.total_s:.2f}s denoise_sum={stats.denoise_s:.2f}s")
    print(
        f"  transformer_forward_sum={stats.transformer_forward_s:.2f}s "
        f"({len(stats.transformer_step_ms)} forwards)"
    )
    print(f"  denoise step0={s0:.1f}ms steps1+_avg={s_rest:.1f}ms")
    print(f"  transformer step0={tf0:.1f}ms steps1+_avg={tf_rest:.1f}ms")
    print(f"  peak_reserved_mb={stats.peak_reserved_mb}")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
