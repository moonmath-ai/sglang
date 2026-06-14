# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

LTX2_LITELINEAR_DENSE_CONFIG_DIR = (
    Path(__file__).resolve().parent / "litelinear_ltx2_dense_config"
)
DEFAULT_LTX2_SNAPSHOT = Path(
    "/mnt/fs/huggingface_cache/hub/models--Lightricks--LTX-2/snapshots/"
    "47da56e2ad66ce4125a9922b4a8826bf407f9d0a"
)
DEFAULT_DISTILLED_TRANSFORMER = DEFAULT_LTX2_SNAPSHOT / "ltx-2-19b-distilled.safetensors"
LTX2_DEFAULT_NEGATIVE_PROMPT = (
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
# LiteLinear README / asset filenames (ltx-2-19b-distilled, f72, per-prompt seeds).
MOONMATH_BENCH_PROMPTS: tuple[dict[str, str | int], ...] = (
    {
        "id": "p001",
        "seed": 486307,
        "prompt": (
            "A single water droplet falls from a height, moving in slow motion through "
            "the air before landing on a still surface and creating ripples."
        ),
    },
    {
        "id": "p002",
        "seed": 789012,
        "prompt": (
            "A man in a sleek modern jetpack flying upwards through a futuristic city, "
            "camera tracking from below as he ascends between glass towers."
        ),
    },
    {
        "id": "p003",
        "seed": 650048,
        "prompt": (
            "Two anthropomorphic cats boxing in a well-lit arena, trading punches in a "
            "cinematic wide shot with dynamic camera movement."
        ),
    },
    {
        "id": "p004",
        "seed": 960015,
        "prompt": (
            "A serene view of the banks of the Rhine river, showing calm water, distant "
            "boats, and soft evening light over the shoreline."
        ),
    },
    {
        "id": "p005",
        "seed": 536857,
        "prompt": (
            "A dramatic underwater scene featuring a person swimming through clear blue "
            "water with light rays filtering down from the surface."
        ),
    },
)
MOONMATH_LITELINEAR_DIR = Path("/tmp/litelinear_moonmath_distilled")


def ensure_moonmath_litelinear_dir(distilled_path: str | Path) -> Path:
    """Bundle LiteLinear config.json + distilled safetensors in one directory."""
    bundle_dir = MOONMATH_LITELINEAR_DIR
    bundle_dir.mkdir(parents=True, exist_ok=True)
    config_dst = bundle_dir / "config.json"
    if not config_dst.exists():
        shutil.copy(LTX2_LITELINEAR_DENSE_CONFIG_DIR / "config.json", config_dst)
    weights_link = bundle_dir / "ltx-2-19b-distilled.safetensors"
    resolved = Path(distilled_path).resolve()
    if weights_link.is_symlink() and weights_link.resolve() != resolved:
        weights_link.unlink()
    if not weights_link.exists():
        weights_link.symlink_to(resolved)
    return bundle_dir


def resolve_transformer_weights_path(
    mode: str, args, mode_spec: dict[str, str | Path | None] | None
) -> str | Path | None:
    if mode == "litelinear" and args.workload == "moonmath":
        distilled = args.transformer_weights_path or str(DEFAULT_DISTILLED_TRANSFORMER)
        return ensure_moonmath_litelinear_dir(distilled)
    if args.transformer_weights_path:
        return args.transformer_weights_path
    if mode_spec is not None:
        return mode_spec.get("transformer_weights_path")
    return None


WORKLOAD_PRESETS: dict[str, dict[str, int | str | float | None]] = {
    "small": {
        "height": 512,
        "width": 768,
        "num_frames": 25,
        "num_inference_steps": 30,
        "guidance_scale": 4.0,
        "transformer_weights_path": None,
        "negative_prompt": None,
    },
    "moonmath": {
        "height": 704,
        "width": 1216,
        "num_frames": 72,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "transformer_weights_path": str(DEFAULT_DISTILLED_TRANSFORMER),
        "negative_prompt": LTX2_DEFAULT_NEGATIVE_PROMPT,
    },
}

# Modes runnable on CUDA (H100). NPU/ROCm-only methods are listed for reference.
QUANT_MODES: dict[str, dict[str, str | Path | None]] = {
    "default": {},
    "litelinear": {
        "quantization": "litelinear",
        "transformer_weights_path": LTX2_LITELINEAR_DENSE_CONFIG_DIR,
    },
    "fp8_online": {"quantization": "fp8"},
    "modelopt_fp8": {
        "quantization": "modelopt_fp8",
        "transformer_weights_path": DEFAULT_LTX2_SNAPSHOT
        / "ltx-2-19b-dev-fp8.safetensors",
    },
    "modelopt_fp4": {
        "quantization": "modelopt_fp4",
        "transformer_weights_path": DEFAULT_LTX2_SNAPSHOT
        / "ltx-2-19b-dev-fp4.safetensors",
    },
    "mxfp4_online": {"quantization": "mxfp4"},
    "mxfp8": {"quantization": "mxfp8"},
    "mxfp4_npu": {"quantization": "mxfp4_npu"},
    "modelslim": {"quantization": "modelslim"},
}
CUDA_QUANT_MODES = (
    "default",
    "litelinear",
    "fp8_online",
    "modelopt_fp8",
    "modelopt_fp4",
)


@dataclass(frozen=True)
class RunResult:
    mode: str
    prompt_id: str
    total_duration_ms: float | None
    denoise_ms: float | None
    transformer_forward_ms: float | None
    peak_reserved_mb: float | None
    transformer_size_gb: float | None
    perf_path: Path
    ok: bool
    error: str = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run an LTX video generation demo benchmark comparing default "
            "linear layers against available SGLang quantization methods."
        )
    )
    parser.add_argument(
        "--model-path",
        default=(
            str(DEFAULT_LTX2_SNAPSHOT)
            if DEFAULT_LTX2_SNAPSHOT.is_dir()
            else "Lightricks/LTX-2"
        ),
    )
    parser.add_argument(
        "--modes",
        default=",".join(CUDA_QUANT_MODES),
        help=(
            "Comma-separated modes to run. Built-ins: " + ", ".join(QUANT_MODES.keys())
        ),
    )
    parser.add_argument("--pipeline-class-name", default="LTX2Pipeline")
    parser.add_argument(
        "--workload",
        choices=tuple(WORKLOAD_PRESETS.keys()),
        default="small",
        help="Preset workload. moonmath matches LiteLinear public LTX-2 bench.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Override prompt text (ignored when --prompt-id all on moonmath workload).",
    )
    parser.add_argument(
        "--prompt-id",
        default="p001",
        help="Moonmath prompt id (p001..p005) or 'all' to run every bench prompt.",
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--transformer-weights-path",
        default=None,
        help="Optional single-file transformer checkpoint (e.g. distilled safetensors).",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--result-dir",
        default="/tmp/sglang_litelinear_ltx_demo",
        help="Directory for perf dumps and generated media.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default="/tmp/sglang_litelinear_ltx_hf_cache",
        help="HF cache dir for model downloads. Set empty to keep env.",
    )
    parser.add_argument(
        "--sglang-cli",
        default=None,
        help=(
            "SGLang CLI executable to invoke. By default, use the current "
            "Python interpreter and call sglang.cli.main."
        ),
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to `sglang generate`. Repeat as needed.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download the model snapshot and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


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
    if args.guidance_scale is None:
        args.guidance_scale = float(preset["guidance_scale"])
    if args.negative_prompt is None and preset.get("negative_prompt"):
        args.negative_prompt = str(preset["negative_prompt"])
    if args.transformer_weights_path is None and preset.get("transformer_weights_path"):
        args.transformer_weights_path = str(preset["transformer_weights_path"])
    if args.prompt is None and args.prompt_id != "all":
        match = next(
            (item for item in MOONMATH_BENCH_PROMPTS if item["id"] == args.prompt_id),
            None,
        )
        if match is not None:
            args.prompt = str(match["prompt"])
            if args.seed is None:
                args.seed = int(match["seed"])
    if args.prompt is None:
        args.prompt = (
            "A quiet coastal town at sunrise, fishing boats moving through "
            "golden mist, cinematic camera movement"
        )
    if args.seed is None:
        args.seed = 42


def iter_prompt_jobs(args) -> list[tuple[str, str, int]]:
    if args.prompt_id == "all":
        return [
            (str(item["id"]), str(item["prompt"]), int(item["seed"]))
            for item in MOONMATH_BENCH_PROMPTS
        ]
    return [(args.prompt_id, args.prompt, int(args.seed))]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def download_model(model_path: str, cache_dir: str | None):
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_path,
        cache_dir=cache_dir or None,
        local_files_only=False,
    )


def build_env(args) -> dict[str, str]:
    env = os.environ.copy()
    repo_python = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = (
        repo_python
        if not env.get("PYTHONPATH")
        else f"{repo_python}{os.pathsep}{env['PYTHONPATH']}"
    )
    if args.hf_cache_dir:
        env["HF_HOME"] = args.hf_cache_dir
        env["HF_HUB_CACHE"] = args.hf_cache_dir
    return env


def build_command(args, mode: str, perf_path: Path, output_path: Path) -> list[str]:
    cmd = (
        [args.sglang_cli]
        if args.sglang_cli
        else [sys.executable, "-c", "from sglang.cli.main import main; main()"]
    )
    cmd.extend(
        [
            "generate",
            "--model-path",
            args.model_path,
            "--pipeline-class-name",
            args.pipeline_class_name,
            "--prompt",
            args.prompt,
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--num-frames",
            str(args.num_frames),
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--seed",
            str(args.seed),
            "--guidance-scale",
            str(args.guidance_scale),
            "--num-gpus",
            str(args.num_gpus),
            "--perf-dump-path",
            str(perf_path),
            "--output-file-path",
            str(output_path),
        ]
    )
    mode_spec = QUANT_MODES.get(mode)
    if mode_spec is None:
        raise ValueError(f"Unknown mode {mode!r}. Choose from: {sorted(QUANT_MODES)}")
    if mode_spec.get("quantization"):
        cmd.extend(["--quantization", str(mode_spec["quantization"])])
    weights_path = resolve_transformer_weights_path(mode, args, mode_spec)
    if weights_path:
        cmd.extend(["--transformer-weights-path", str(weights_path)])
    if args.negative_prompt:
        cmd.extend(["--negative-prompt", args.negative_prompt])
    cmd.extend(args.extra_arg)
    return cmd


def read_perf(
    path: Path,
) -> tuple[float | None, float | None, float | None, float | None]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    total_duration_ms = data.get("total_duration_ms")
    denoise_steps = data.get("denoise_steps_ms") or []
    denoise_ms = None
    if denoise_steps:
        denoise_ms = sum(float(item["duration_ms"]) for item in denoise_steps)

    peak_reserved_mb = None
    for snapshot in (data.get("memory_checkpoints") or {}).values():
        value = snapshot.get("peak_reserved_mb")
        if value is not None:
            peak_reserved_mb = max(peak_reserved_mb or 0.0, float(value))

    transformer_forward_ms = None
    transformer_steps = data.get("transformer_forward_steps_ms") or []
    if transformer_steps:
        transformer_forward_ms = sum(
            float(item["duration_ms"]) for item in transformer_steps
        )

    return total_duration_ms, denoise_ms, transformer_forward_ms, peak_reserved_mb


def parse_transformer_size_gb(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    match = re.search(
        r"Loaded transformer:.*model size:\s*([0-9.]+)\s*GB",
        log_path.read_text(encoding="utf-8", errors="replace"),
    )
    return float(match.group(1)) if match else None


def run_mode(
    args,
    mode: str,
    prompt_id: str,
    prompt: str,
    seed: int,
    result_dir: Path,
    env: dict[str, str],
) -> RunResult:
    run_args = argparse.Namespace(**vars(args))
    run_args.prompt = prompt
    run_args.seed = seed
    tag = safe_name(f"{mode}_{prompt_id}")
    perf_path = result_dir / f"{tag}.perf.json"
    output_path = result_dir / f"{tag}.mp4"
    log_path = result_dir / f"{tag}.log"
    cmd = build_command(run_args, mode, perf_path, output_path)
    print("\n$ " + " ".join(cmd), flush=True)
    if args.dry_run:
        return RunResult(
            mode,
            prompt_id,
            None,
            None,
            None,
            None,
            None,
            perf_path,
            ok=True,
        )

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        returncode = process.wait()

    if returncode != 0:
        return RunResult(
            mode,
            prompt_id,
            None,
            None,
            None,
            None,
            None,
            perf_path,
            ok=False,
            error=f"exit code {returncode}",
        )

    if not perf_path.exists():
        return RunResult(
            mode,
            prompt_id,
            None,
            None,
            None,
            None,
            None,
            perf_path,
            ok=False,
            error=f"missing perf dump: {perf_path}",
        )

    total_duration_ms, denoise_ms, transformer_forward_ms, peak_reserved_mb = read_perf(
        perf_path
    )
    transformer_size_gb = parse_transformer_size_gb(log_path)
    return RunResult(
        mode,
        prompt_id,
        total_duration_ms,
        denoise_ms,
        transformer_forward_ms,
        peak_reserved_mb,
        transformer_size_gb,
        perf_path,
        ok=True,
    )


def format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.2f}"


def format_change(default: float | None, value: float | None) -> str:
    if default is None or value is None or value == 0:
        return "n/a"
    speedup = default / value
    faster = (default - value) / default * 100.0
    return f"{speedup:.3f}x ({faster:+.2f}%)"


def format_gb(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def summarize_mode(results: list[RunResult], mode: str) -> dict[str, float | None]:
    rows = [item for item in results if item.mode == mode and item.ok]
    if not rows:
        return {
            "transformer_forward_ms": None,
            "denoise_ms": None,
            "total_ms": None,
        }

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    return {
        "transformer_forward_ms": mean(
            [item.transformer_forward_ms for item in rows if item.transformer_forward_ms]
        )
        if any(item.transformer_forward_ms for item in rows)
        else None,
        "denoise_ms": mean([item.denoise_ms for item in rows if item.denoise_ms])
        if any(item.denoise_ms for item in rows)
        else None,
        "total_ms": mean(
            [item.total_duration_ms for item in rows if item.total_duration_ms]
        )
        if any(item.total_duration_ms for item in rows)
        else None,
    }


def print_table(results: list[RunResult]):
    baseline = summarize_mode(results, "default")
    default_transformer_gb = next(
        (
            item.transformer_size_gb
            for item in results
            if item.mode == "default" and item.ok and item.transformer_size_gb
        ),
        None,
    )

    print("\nLTX Quantization Comparison")
    print(
        "| Mode | Prompt | OK | transformer GB | transformer ms | denoise ms | "
        "total ms | peak reserved MB | transformer vs baseline |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            "| "
            f"{result.mode} | "
            f"{result.prompt_id} | "
            f"{'yes' if result.ok else 'no'} | "
            f"{format_gb(result.transformer_size_gb)} | "
            f"{format_ms(result.transformer_forward_ms)} | "
            f"{format_ms(result.denoise_ms)} | "
            f"{format_ms(result.total_duration_ms)} | "
            f"{format_ms(result.peak_reserved_mb)} | "
            f"{format_change(baseline['transformer_forward_ms'], result.transformer_forward_ms)} |"
        )
        if result.error:
            print(
                f"# {result.mode}/{result.prompt_id} error: {result.error}",
                file=sys.stderr,
            )

    print("\nMode means (apples-to-apples vs LiteLinear 'Transformer Mean')")
    print("| Mode | mean transformer s | mean denoise s | mean total s | transformer GB |")
    print("| --- | ---: | ---: | ---: | ---: |")
    modes = sorted({item.mode for item in results})
    for mode in modes:
        summary = summarize_mode(results, mode)
        gb = next(
            (
                item.transformer_size_gb
                for item in results
                if item.mode == mode and item.ok and item.transformer_size_gb
            ),
            None,
        )
        tf_s = (
            summary["transformer_forward_ms"] / 1000.0
            if summary["transformer_forward_ms"]
            else None
        )
        den_s = summary["denoise_ms"] / 1000.0 if summary["denoise_ms"] else None
        tot_s = summary["total_ms"] / 1000.0 if summary["total_ms"] else None
        tf_cell = f"{tf_s:.3f}" if tf_s is not None else "n/a"
        den_cell = f"{den_s:.3f}" if den_s is not None else "n/a"
        tot_cell = f"{tot_s:.3f}" if tot_s is not None else "n/a"
        print(f"| {mode} | {tf_cell} | {den_cell} | {tot_cell} | {format_gb(gb)} |")

    if baseline["transformer_forward_ms"] and summarize_mode(results, "litelinear")[
        "transformer_forward_ms"
    ]:
        base_s = baseline["transformer_forward_ms"] / 1000.0
        ll_s = (
            summarize_mode(results, "litelinear")["transformer_forward_ms"] / 1000.0
        )
        faster = (base_s - ll_s) / base_s * 100.0
        print(
            f"\nSGLang transformer speedup (litelinear vs default): "
            f"{base_s:.3f}s -> {ll_s:.3f}s ({faster:+.2f}% faster)"
        )
        print("LiteLinear README reference (H200): 4.520s -> 3.500s (22.57% faster)")
    if default_transformer_gb:
        print(f"Baseline transformer size: {default_transformer_gb:.2f} GB")


def main():
    args = parse_args()
    resolve_workload(args)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(args)

    if args.hf_cache_dir:
        Path(args.hf_cache_dir).mkdir(parents=True, exist_ok=True)

    if args.download_only:
        path = download_model(args.model_path, args.hf_cache_dir or None)
        print(path)
        return

    unknown = [mode for mode in args.modes.split(",") if mode not in QUANT_MODES]
    if unknown:
        raise SystemExit(
            f"Unknown mode(s): {unknown}. Choose from: {sorted(QUANT_MODES)}"
        )

    prompt_jobs = iter_prompt_jobs(args)
    results: list[RunResult] = []
    for mode in args.modes.split(","):
        mode = mode.strip()
        if not mode:
            continue
        for prompt_id, prompt, seed in prompt_jobs:
            results.append(
                run_mode(args, mode, prompt_id, prompt, seed, result_dir, env)
            )
    print_table(results)

    if not any(result.ok for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
