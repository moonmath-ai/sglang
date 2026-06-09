# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen


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
    parser.add_argument(
        "--quality-eval",
        choices=["none", "gsm8k"],
        default="none",
        help="Run an optional SGLang quality eval after speed benchmarks.",
    )
    parser.add_argument(
        "--quality-load-format",
        default="auto",
        help=(
            "Load format for quality eval servers. Use real weights for useful "
            "quality numbers."
        ),
    )
    parser.add_argument("--quality-host", default="127.0.0.1")
    parser.add_argument("--quality-port", type=int, default=30000)
    parser.add_argument("--quality-timeout", type=int, default=900)
    parser.add_argument("--quality-num-examples", type=int, default=32)
    parser.add_argument("--quality-num-threads", type=int, default=32)
    parser.add_argument("--quality-num-shots", type=int, default=5)
    parser.add_argument("--quality-max-tokens", type=int, default=512)
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


def build_server_command(args, mode: str, port: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--load-format",
        args.quality_load_format,
        "--dtype",
        args.dtype,
        "--host",
        args.quality_host,
        "--port",
        str(port),
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
    ]
    quantization = mode_to_quantization(mode)
    if quantization is not None:
        cmd.extend(["--quantization", quantization])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def wait_for_server(base_url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with urlopen(f"{base_url}/health", timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, URLError) as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"Server at {base_url} did not become healthy: {last_error}")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def run_gsm8k_quality(
    args, mode: str, env: dict[str, str], result_file: Path, port: int
) -> dict:
    base_url = f"http://{args.quality_host}:{port}"
    server_cmd = build_server_command(args, mode, port)
    eval_code = f"""
import json
from types import SimpleNamespace
from sglang.test.run_eval import run_eval
args = SimpleNamespace(
    base_url={base_url!r},
    host={args.quality_host!r},
    port={port},
    model={args.model_path!r},
    eval_name="gsm8k",
    api="completion",
    max_tokens={args.quality_max_tokens},
    temperature=0.0,
    top_p=1.0,
    num_examples={args.quality_num_examples},
    num_threads={args.quality_num_threads},
    num_shots={args.quality_num_shots},
    repeat=1,
)
metrics = run_eval(args)
with open({str(result_file)!r}, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, sort_keys=True))
"""
    eval_cmd = [sys.executable, "-c", eval_code]

    print("\n$ " + " ".join(server_cmd), flush=True)
    server_log_file = result_file.with_suffix(".server.log")
    server_log = server_log_file.open("w", encoding="utf-8")
    server_proc = subprocess.Popen(
        server_cmd,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        wait_for_server(base_url, args.quality_timeout)
        run_command(eval_cmd, env)
    finally:
        stop_server(server_proc)
        server_log.close()
        print(f"Server log: {server_log_file}")

    with result_file.open(encoding="utf-8") as f:
        return json.load(f)


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


def pct(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return (new / old - 1.0) * 100.0


def print_summary(
    rows: Iterable[tuple[DemoCase, dict[str, dict]]],
    modes: list[str],
) -> None:
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


def print_quality_summary(results: dict[str, dict], modes: list[str]) -> None:
    if not results:
        return

    print("\nLiteLinear Quality Summary")
    print(
        "| Eval | Metric | Mode | Default | Mode value | Change |\n"
        "| --- | --- | --- | ---: | ---: | ---: |"
    )
    default = results["default"]
    default_score = default.get("score")
    default_latency = default.get("latency")
    for mode in modes:
        if mode == "default":
            continue
        mode_result = results[mode]
        mode_score = mode_result.get("score")
        score_change = pct(mode_score, default_score)
        print(
            f"| gsm8k | score | `{mode}` | {fmt_float(default_score)} | "
            f"{fmt_float(mode_score)} | {fmt_float(score_change, '%')} |"
        )
        mode_latency = mode_result.get("latency")
        if default_latency is not None and mode_latency is not None:
            latency_change = pct(mode_latency, default_latency)
            print(
                f"| gsm8k | latency s | `{mode}` | {fmt_float(default_latency)} | "
                f"{fmt_float(mode_latency)} | {fmt_float(latency_change, '%')} |"
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

    if args.quality_eval == "gsm8k":
        quality_results = {}
        for i, mode in enumerate(modes):
            quality_file = result_dir / f"{safe_mode_name(mode)}_gsm8k_quality.json"
            quality_file.unlink(missing_ok=True)
            quality_port = args.quality_port + i
            if args.dry_run:
                print(" ".join(build_server_command(args, mode, quality_port)))
                continue
            quality_results[mode] = run_gsm8k_quality(
                args,
                mode,
                env,
                quality_file,
                quality_port,
            )
        print_quality_summary(quality_results, modes)


if __name__ == "__main__":
    main()
