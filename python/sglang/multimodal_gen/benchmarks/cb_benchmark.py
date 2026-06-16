# SPDX-License-Identifier: Apache-2.0
"""Unified Continuous Batching Benchmark Script for Diffusion Serving.

Combines three benchmark scenarios to evaluate scheduling performance:
1. Concurrency (Closed-loop): Fires requests at a fixed concurrency.
2. Mixed-steps: Fires requests with heterogeneous step sizes (SHORT vs. LONG).
3. Open-loop: Fires requests at Poisson-distributed arrival times.

Examples:
    # 1. Standard Concurrency Load Test
    python -m sglang.multimodal_gen.benchmarks.cb_benchmark --mode concurrency --concurrency 8 --total 32

    # 2. Mixed Step Size Test (Diverse step counts)
    python -m sglang.multimodal_gen.benchmarks.cb_benchmark --mode mixed --concurrency 8 --total 32

    # 3. Open-loop Poisson-arrival Test
    python -m sglang.multimodal_gen.benchmarks.cb_benchmark --mode openloop --rate 4.0 --total 40
"""

import argparse
import concurrent.futures
import json
import math
import statistics
import threading
import time
import urllib.request

STEP_SPREAD = [8, 12, 16, 20, 28, 36, 50]


def fire_request(
    url: str,
    idx: int,
    steps: int,
    size: int,
    guidance_scale: float,
    negative_prompt: str | None,
    timeout: float,
) -> tuple[float, bool]:
    """Execute a single HTTP image generation request and return (latency, success)."""
    body = json.dumps(
        {
            "prompt": f"a detailed photo of object {idx}, scene {idx % 8}",
            "negative_prompt": negative_prompt,
            "width": size,
            "height": size,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "n": 1,
            "seed": 10000 + idx,
            "response_format": "b64_json",
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/images/generations",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        return time.time() - start, True
    except Exception as exc:  # noqa: BLE001
        print(f"  req{idx} (steps={steps}) failed: {exc}")
        return time.time() - start, False


def run_concurrency(args: argparse.Namespace) -> None:
    print(f"Running concurrency (closed-loop) test: concurrency={args.concurrency}, total={args.total}")
    latencies: list[float] = []
    ok = 0
    wall_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                fire_request,
                args.url,
                i,
                args.steps,
                args.size,
                args.guidance_scale,
                args.negative_prompt,
                args.timeout,
            )
            for i in range(args.total)
        ]
        for fut in concurrent.futures.as_completed(futures):
            latency, success = fut.result()
            latencies.append(latency)
            ok += int(success)

    wall = time.time() - wall_start
    latencies.sort()

    def pct(q: float) -> float:
        return latencies[min(len(latencies) - 1, int(q / 100 * len(latencies)))] if latencies else float("nan")

    print(
        f"[{args.label}] concurrency={args.concurrency} total={args.total} ok={ok}/{args.total} "
        f"wall={wall:.2f}s throughput={ok / wall:.3f} req/s | "
        f"latency mean={statistics.mean(latencies):.2f}s p50={pct(50):.2f} "
        f"p90={pct(90):.2f} p99={pct(99):.2f}"
    )


def run_mixed(args: argparse.Namespace) -> None:
    print(f"Running mixed-step test: concurrency={args.concurrency}, total={args.total}")
    if args.unique:
        plan = [(i, args.step_base + i) for i in range(args.total)]
    else:
        plan = [(i, STEP_SPREAD[i % len(STEP_SPREAD)]) for i in range(args.total)]

    results: list[tuple[int, float, bool]] = []
    wall_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                fire_request,
                args.url,
                idx,
                steps,
                args.size,
                args.guidance_scale,
                args.negative_prompt,
                args.timeout,
            )
            for idx, steps in plan
        ]
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            # Retain original step count mapped to the completed task
            steps = plan[i][1]
            latency, success = fut.result()
            results.append((steps, latency, success))

    wall = time.time() - wall_start
    ok = sum(1 for _, _, s in results if s)
    short = sorted(dt for st, dt, s in results if s and st <= 16)
    long = sorted(dt for st, dt, s in results if s and st >= 36)

    def med(xs: list[float]) -> float:
        return statistics.median(xs) if xs else float("nan")

    def mx(xs: list[float]) -> float:
        return max(xs) if xs else float("nan")

    print(
        f"[{args.label}] mixed-steps conc={args.concurrency} total={args.total} ok={ok}/{args.total} "
        f"wall={wall:.2f}s throughput={ok / wall:.3f} req/s | "
        f"SHORT(<=16) n={len(short)} p50={med(short):.2f}s max={mx(short):.2f}s | "
        f"LONG(>=36) n={len(long)} p50={med(long):.2f}s max={mx(long):.2f}s"
    )


def run_openloop(args: argparse.Namespace) -> None:
    print(f"Running open-loop (Poisson-arrival) test: rate={args.rate}/s, total={args.total}")
    # Deterministic Poisson inter-arrival gaps (exponential) using LCG
    state = args.seed & 0xFFFFFFFF

    def _u():
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return (state + 1) / (0x7FFFFFFF + 2)

    gaps = [-math.log(_u()) / args.rate for _ in range(args.total)]

    out: list[tuple[float, bool]] = []
    threads = []
    t0 = time.time()

    def _worker(idx):
        lat, success = fire_request(
            args.url,
            idx,
            args.steps,
            args.size,
            args.guidance_scale,
            args.negative_prompt,
            args.timeout,
        )
        out.append((lat, success))

    for idx, gap in enumerate(gaps):
        target = t0 + sum(gaps[: idx + 1])
        now = time.time()
        if target > now:
            time.sleep(target - now)
        th = threading.Thread(target=_worker, args=(idx,))
        th.start()
        threads.append(th)

    for th in threads:
        th.join()

    wall = time.time() - t0
    lat = sorted(dt for dt, ok in out if ok)
    okn = sum(1 for _, ok in out if ok)

    def pct(q: float) -> float:
        return lat[min(len(lat) - 1, int(q / 100 * len(lat)))] if lat else float("nan")

    print(
        f"[{args.label}] open-loop rate={args.rate}/s total={args.total} ok={okn}/{args.total} "
        f"achieved={okn / wall:.2f} req/s | e2e latency "
        f"p50={pct(50):.2f}s p90={pct(90):.2f}s p99={pct(99):.2f}s "
        f"mean={statistics.mean(lat):.2f}s"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["concurrency", "mixed", "openloop"],
        default="concurrency",
        help="Benchmark scenario to run",
    )
    p.add_argument("--url", default="http://127.0.0.1:30000", help="Model server endpoint")
    p.add_argument("--label", default="benchmark", help="Benchmark run label prefix")
    p.add_argument("--total", type=int, default=32, help="Total requests to fire")
    p.add_argument("--size", type=int, default=512, help="Image width/height resolution")
    p.add_argument("--steps", type=int, default=4, help="Inference steps for uniform/concurrency/openloop modes")
    p.add_argument("--guidance-scale", type=float, default=0.0, help="Guidance scale parameter")
    p.add_argument("--negative-prompt", default=None, help="Optional negative prompt")
    p.add_argument("--timeout", type=float, default=600.0, help="HTTP request timeout in seconds")
    p.add_argument("--no-warmup", action="store_true", help="Skip single warmup request execution")

    # Concurrency / Mixed modes
    p.add_argument("--concurrency", type=int, default=8, help="Concurrency level for closed-loop execution")

    # Mixed-steps specific
    p.add_argument("--unique", action="store_true", help="Mixed-steps: unique step sizes per request")
    p.add_argument("--step-base", type=int, default=10, help="Mixed-steps: step base value for unique size ranges")

    # Open-loop specific
    p.add_argument("--rate", type=float, default=4.0, help="Open-loop: arrival target rate (arrivals/sec)")
    p.add_argument("--seed", type=int, default=1234, help="Open-loop: seed for Poisson arrival generation")

    args = p.parse_args()

    # Dynamic defaults adjustment if user did not specify total
    if args.total == 32:
        if args.mode == "mixed":
            args.total = 56
        elif args.mode == "openloop":
            args.total = 80

    if not args.no_warmup:
        print("Running warmup request...")
        fire_request(
            args.url,
            -1,
            16 if args.mode == "mixed" else args.steps,
            args.size,
            args.guidance_scale,
            args.negative_prompt,
            args.timeout,
        )

    if args.mode == "concurrency":
        run_concurrency(args)
    elif args.mode == "mixed":
        run_mixed(args)
    elif args.mode == "openloop":
        run_openloop(args)


if __name__ == "__main__":
    main()
