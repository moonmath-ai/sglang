# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunResult:
    mode: str
    total_duration_ms: float | None
    denoise_ms: float | None
    peak_reserved_mb: float | None
    perf_path: Path
    ok: bool
    error: str = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run an LTX video generation demo benchmark with SGLang default "
            "linear layers and with --quantization litelinear."
        )
    )
    parser.add_argument("--model-path", default="Lightricks/LTX-2")
    parser.add_argument("--pipeline-class-name", default="LTX2Pipeline")
    parser.add_argument(
        "--prompt",
        default=(
            "A quiet coastal town at sunrise, fishing boats moving through "
            "golden mist, cinematic camera movement"
        ),
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
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
            "--num-gpus",
            str(args.num_gpus),
            "--perf-dump-path",
            str(perf_path),
            "--output-file-path",
            str(output_path),
        ]
    )
    if mode == "litelinear":
        cmd.extend(["--quantization", "litelinear"])
    cmd.extend(args.extra_arg)
    return cmd


def read_perf(path: Path) -> tuple[float | None, float | None, float | None]:
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

    return total_duration_ms, denoise_ms, peak_reserved_mb


def run_mode(args, mode: str, result_dir: Path, env: dict[str, str]) -> RunResult:
    perf_path = result_dir / f"{safe_name(mode)}.perf.json"
    output_path = result_dir / f"{safe_name(mode)}.mp4"
    log_path = result_dir / f"{safe_name(mode)}.log"
    cmd = build_command(args, mode, perf_path, output_path)
    print("\n$ " + " ".join(cmd), flush=True)
    if args.dry_run:
        return RunResult(mode, None, None, None, perf_path, ok=True)

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
            None,
            None,
            None,
            perf_path,
            ok=False,
            error=f"missing perf dump: {perf_path}",
        )

    total_duration_ms, denoise_ms, peak_reserved_mb = read_perf(perf_path)
    return RunResult(
        mode,
        total_duration_ms,
        denoise_ms,
        peak_reserved_mb,
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


def print_table(results: list[RunResult]):
    default = next((item for item in results if item.mode == "default"), None)
    default_total = default.total_duration_ms if default else None
    default_denoise = default.denoise_ms if default else None

    print("\nLTX LiteLinear Demo Summary")
    print(
        "| Mode | OK | total ms | denoise ms | peak reserved MB | "
        "total change | denoise change |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            "| "
            f"{result.mode} | "
            f"{'yes' if result.ok else 'no'} | "
            f"{format_ms(result.total_duration_ms)} | "
            f"{format_ms(result.denoise_ms)} | "
            f"{format_ms(result.peak_reserved_mb)} | "
            f"{format_change(default_total, result.total_duration_ms)} | "
            f"{format_change(default_denoise, result.denoise_ms)} |"
        )
        if result.error:
            print(f"# {result.mode} error: {result.error}", file=sys.stderr)


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(args)

    if args.hf_cache_dir:
        Path(args.hf_cache_dir).mkdir(parents=True, exist_ok=True)

    if args.download_only:
        path = download_model(args.model_path, args.hf_cache_dir or None)
        print(path)
        return

    results = [
        run_mode(args, "default", result_dir, env),
        run_mode(args, "litelinear", result_dir, env),
    ]
    print_table(results)

    if not all(result.ok for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
