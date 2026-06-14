# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: native LiteLinear vs SGLang ColumnParallelLinear wrapper."""

from __future__ import annotations

import argparse
import os
import time

import torch

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29541")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

from sglang.multimodal_gen.runtime.layers.linear import ColumnParallelLinear
from sglang.multimodal_gen.runtime.layers.quantization.litelinear import (
    LiteLinearConfig,
    LiteLinearMethod,
)
from sglang.multimodal_gen.runtime.distributed import (
    maybe_init_distributed_environment_and_model_parallel,
)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--mult", type=int, default=4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    inner = args.dim * args.mult
    x = torch.randn(args.batch, args.seq, args.dim, device=device, dtype=dtype)

    # Native diffusers-style LiteLinear (proj_in shape).
    from lite_linear import LiteLinear

    native = LiteLinear(
        args.dim, inner, rank=args.rank, bias=True, device=device, dtype=dtype
    )
    with torch.no_grad():
        native.weight.copy_(torch.randn_like(native.weight) * 0.02)
        native.bias.zero_()
    native.materialize_from_weight()
    native.eval()

    # SGLang ColumnParallelLinear + LiteLinearMethod.
    quant_cfg = LiteLinearConfig(
        rank=args.rank,
        target_patterns=[r"proj_in$"],
    )
    sgl_layer = ColumnParallelLinear(
        args.dim,
        inner,
        bias=True,
        gather_output=False,
        params_dtype=dtype,
        quant_config=quant_cfg,
        prefix="transformer_blocks.0.ff.proj_in",
    ).to(device)
    with torch.no_grad():
        sgl_layer.weight.copy_(native.weight)
        sgl_layer.bias.zero_()
    LiteLinearMethod(quant_cfg, "transformer_blocks.0.ff.proj_in").process_weights_after_loading(
        sgl_layer
    )
    sgl_layer.eval()

    # BF16 baseline through same wrapper (no LiteLinear).
    bf16_layer = ColumnParallelLinear(
        args.dim,
        inner,
        bias=True,
        gather_output=False,
        params_dtype=dtype,
        quant_config=None,
        prefix="bf16",
    ).to(device)
    with torch.no_grad():
        bf16_layer.weight.copy_(native.weight)
        bf16_layer.bias.zero_()
    bf16_layer.eval()

    def native_fwd():
        return native(x)

    def sgl_fwd():
        out, _ = sgl_layer(x)
        return out

    def bf16_fwd():
        out, _ = bf16_layer(x)
        return out

    def sgl_apply_only():
        return sgl_layer.quant_method.apply(sgl_layer, x, sgl_layer.bias)

    # Warmup includes first-kernel JIT for all paths.
    _bench(native_fwd, warmup=2, iters=1)
    _bench(sgl_fwd, warmup=2, iters=1)
    _bench(bf16_fwd, warmup=2, iters=1)

    results = {
        "native_litelinear_ms": _bench(native_fwd, args.warmup, args.iters),
        "sglang_litelinear_fwd_ms": _bench(sgl_fwd, args.warmup, args.iters),
        "sglang_litelinear_apply_ms": _bench(sgl_apply_only, args.warmup, args.iters),
        "sglang_bf16_fwd_ms": _bench(bf16_fwd, args.warmup, args.iters),
    }

    shape = f"B={args.batch} S={args.seq} dim={args.dim}->{inner} rank={args.rank}"
    print(shape)
    for k, v in results.items():
        print(f"  {k}: {v:.3f} ms")

    overhead = results["sglang_litelinear_fwd_ms"] - results["native_litelinear_ms"]
    wrapper_pct = 100.0 * overhead / results["native_litelinear_ms"]
    print(f"  wrapper_overhead_vs_native: {overhead:+.3f} ms ({wrapper_pct:+.2f}%)")

    vs_bf16 = results["sglang_litelinear_fwd_ms"] - results["sglang_bf16_fwd_ms"]
    ll_speedup = 100.0 * (results["sglang_bf16_fwd_ms"] - results["sglang_litelinear_fwd_ms"]) / results["sglang_bf16_fwd_ms"]
    print(f"  litelinear_vs_bf16_in_sglang: {vs_bf16:+.3f} ms ({ll_speedup:+.2f}% faster)")


if __name__ == "__main__":
    main()
