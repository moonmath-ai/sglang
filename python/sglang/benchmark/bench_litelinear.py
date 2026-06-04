# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from typing import Callable

import torch

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.litelinear import LiteLinearConfig


@dataclass(frozen=True)
class BenchResult:
    name: str
    median_ms: float
    min_ms: float
    max_ms: float


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare SGLang default ReplicatedLinear against the optional "
            "LiteLinear quantization path on synthetic FFN projection shapes."
        )
    )
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=14336)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix", default="model.layers.0.mlp.gate_up_proj")
    parser.add_argument("--min-input-size", type=int, default=4096)
    return parser.parse_args()


def make_layer(
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
    quant_config=None,
    prefix: str = "model.layers.0.mlp.gate_up_proj",
):
    layer = ReplicatedLinear(
        in_features,
        out_features,
        bias=False,
        params_dtype=dtype,
        quant_config=quant_config,
        prefix=prefix,
    ).to(device=device)
    return layer.eval()


def copy_weight(dst: ReplicatedLinear, src: ReplicatedLinear):
    with torch.no_grad():
        dst.weight.copy_(src.weight)


def time_cuda(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> BenchResult:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    return BenchResult(
        name="",
        median_ms=statistics.median(timings),
        min_ms=min(timings),
        max_ms=max(timings),
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LiteLinear benchmark requires a CUDA device.")

    dtype = getattr(torch, args.dtype)
    x = torch.randn(args.tokens, args.in_features, dtype=dtype, device=device)

    baseline = make_layer(
        args.in_features,
        args.out_features,
        dtype,
        device,
        prefix=args.prefix,
    )
    with torch.no_grad():
        baseline.weight.normal_(mean=0.0, std=0.02)

    litelinear = make_layer(
        args.in_features,
        args.out_features,
        dtype,
        device,
        quant_config=LiteLinearConfig(
            rank=args.rank,
            min_input_size=args.min_input_size,
        ),
        prefix=args.prefix,
    )
    copy_weight(litelinear, baseline)
    litelinear.quant_method.process_weights_after_loading(litelinear)

    with torch.no_grad():
        baseline_out = baseline(x)[0]
        litelinear_out = litelinear(x)[0]

    mse = torch.mean((baseline_out.float() - litelinear_out.float()) ** 2).item()
    max_abs = torch.max(torch.abs(baseline_out.float() - litelinear_out.float())).item()

    baseline_result = time_cuda(lambda: baseline(x)[0], args.warmup, args.iters)
    litelinear_result = time_cuda(lambda: litelinear(x)[0], args.warmup, args.iters)
    speedup = baseline_result.median_ms / litelinear_result.median_ms

    print(
        f"shape: tokens={args.tokens}, in={args.in_features}, "
        f"out={args.out_features}, dtype={args.dtype}, rank={args.rank}"
    )
    print(f"quality: mse={mse:.6e}, max_abs={max_abs:.6e}")
    print(
        "default:    "
        f"median={baseline_result.median_ms:.3f} ms, "
        f"min={baseline_result.min_ms:.3f}, max={baseline_result.max_ms:.3f}"
    )
    print(
        "litelinear: "
        f"median={litelinear_result.median_ms:.3f} ms, "
        f"min={litelinear_result.min_ms:.3f}, max={litelinear_result.max_ms:.3f}"
    )
    print(f"speedup: {speedup:.3f}x")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
