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
from typing import Iterable


@dataclass(frozen=True)
class DemoCase:
    name: str
    batch_size: int
    input_len: int
    output_len: int


DEFAULT_CASES = {
    "prefill": DemoCase("prefill", batch_size=8, input_len=4096, output_len=1),
    "decode": DemoCase("decode", batch_size=32, input_len=128, output_len=128),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Demo SGLang LLM speed with and without --quantization litelinear. "
            "Defaults use dummy weights so the demo does not download model shards."
        )
    )
    parser.add_argument(
        "--model-path",
        default="NousResearch/Meta-Llama-3-8B-Instruct",
        help="Model config/tokenizer to use. Dummy weights are used by default.",
    )
    parser.add_argument(
        "--load-format",
        default="dummy",
        help="Use 'dummy' for a quick kernel demo, or 'auto' for real weights.",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--case", choices=["all", "prefill", "decode"], default="all")
    parser.add_argument("--prefill-batch-size", type=int, default=8)
    parser.add_argument("--prefill-input-len", type=int, default=4096)
    parser.add_argument("--prefill-output-len", type=int, default=1)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--decode-input-len", type=int, default=128)
    parser.add_argument("--decode-output-len", type=int, default=128)
    parser.add_argument(
        "--result-dir",
        default="/tmp/sglang_litelinear_demo",
        help="Directory for temporary bench_one_batch JSONL files.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default="/tmp/sglang_litelinear_demo_hf_cache",
        help="HF cache dir for config/tokenizer downloads. Set empty to keep env.",
    )
    parser.add_argument(
        "--keep-piecewise-cuda-graph",
        action="store_true",
        help="Do not pass --disable-piecewise-cuda-graph to child benchmarks.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward --trust-remote-code to bench_one_batch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print child commands without running them.",
    )
    parser.add_argument(
        "--extra-quantization",
        nargs="*",
        default=[],
        help=(
            "Extra --quantization modes to compare against default, e.g. "
            "'fp8 w8a8_fp8 w8a8_int8'. Modes that require special checkpoints "
            "or missing dependencies may fail."
        ),
    )
    return parser.parse_args()


def build_cases(args) -> list[DemoCase]:
    cases = {
        "prefill": DemoCase(
            "prefill",
            batch_size=args.prefill_batch_size,
            input_len=args.prefill_input_len,
            output_len=args.prefill_output_len,
        ),
        "decode": DemoCase(
            "decode",
            batch_size=args.decode_batch_size,
            input_len=args.decode_input_len,
            output_len=args.decode_output_len,
        ),
    }
    if args.case == "all":
        return [cases["prefill"], cases["decode"]]
    return [cases[args.case]]


def mode_to_quantization(mode: str) -> str | None:
    if mode == "default":
        return None
    return mode


def safe_mode_name(mode: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", mode)


def build_command(args, case: DemoCase, mode: str, result_file: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_one_batch",
        "--model-path",
        args.model_path,
        "--load-format",
        args.load_format,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(case.batch_size),
        "--input-len",
        str(case.input_len),
        "--output-len",
        str(case.output_len),
        "--grammar-backend",
        "none",
        "--run-name",
        f"{mode}_{case.name}",
        "--result-filename",
        str(result_file),
    ]
    quantization = mode_to_quantization(mode)
    if quantization is not None:
        cmd.extend(["--quantization", quantization])
    if not args.keep_piecewise_cuda_graph:
        cmd.append("--disable-piecewise-cuda-graph")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def child_env(args) -> dict[str, str]:
    env = os.environ.copy()
    if args.hf_cache_dir:
        cache_dir = str(Path(args.hf_cache_dir).expanduser())
        env["HF_HOME"] = cache_dir
        env["HF_HUB_CACHE"] = str(Path(cache_dir) / "hub")
        env["HUGGINGFACE_HUB_CACHE"] = str(Path(cache_dir) / "hub")
        env["TRANSFORMERS_CACHE"] = str(Path(cache_dir) / "hub")
    return env


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def load_last_result(path: Path) -> dict:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No benchmark result was written to {path}")
    return rows[-1]


def fmt_float(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:,.2f}{suffix}"


def pct(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def print_summary(rows: Iterable[tuple[DemoCase, dict[str, dict]]], modes: list[str]) -> None:
    print("\nLiteLinear Demo Summary")
    print(
        "| Case | Metric | Mode | Default | Mode value | Change |\n"
        "| --- | --- | --- | ---: | ---: | ---: |"
    )
    for case, results in rows:
        default = results["default"]
        default_prefill = default.get("prefill_throughput")
        default_decode = default.get("median_decode_throughput")
        default_total = default.get("overall_throughput")

        for mode in modes:
            if mode == "default":
                continue
            mode_result = results[mode]
            mode_prefill = mode_result.get("prefill_throughput")
            print(
                f"| {case.name} | prefill tok/s | `{mode}` | "
                f"{fmt_float(default_prefill)} | {fmt_float(mode_prefill)} | "
                f"{fmt_float(pct(mode_prefill, default_prefill), '%')} |"
            )

            mode_decode = mode_result.get("median_decode_throughput")
            if default_decode is not None and mode_decode is not None:
                print(
                    f"| {case.name} | median decode tok/s | `{mode}` | "
                    f"{fmt_float(default_decode)} | {fmt_float(mode_decode)} | "
                    f"{fmt_float(pct(mode_decode, default_decode), '%')} |"
                )

            mode_total = mode_result.get("overall_throughput")
            print(
                f"| {case.name} | overall tok/s | `{mode}` | "
                f"{fmt_float(default_total)} | {fmt_float(mode_total)} | "
                f"{fmt_float(pct(mode_total, default_total), '%')} |"
            )


def main():
    args = parse_args()
    result_dir = Path(args.result_dir).expanduser()
    result_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(args)
    env = child_env(args)
    summary_rows = []
    modes = ["default", "litelinear", *args.extra_quantization]

    for case in cases:
        result_files = {
            mode: result_dir / f"{safe_mode_name(mode)}_{case.name}.jsonl"
            for mode in modes
        }
        for path in result_files.values():
            path.unlink(missing_ok=True)

        commands = {
            mode: build_command(args, case, mode, result_file)
            for mode, result_file in result_files.items()
        }

        if args.dry_run:
            for cmd in commands.values():
                print(" ".join(cmd))
            continue

        for mode in modes:
            run_command(commands[mode], env)

        summary_rows.append(
            (case, {mode: load_last_result(result_files[mode]) for mode in modes})
        )

    if summary_rows:
        print_summary(summary_rows, modes)


if __name__ == "__main__":
    main()
