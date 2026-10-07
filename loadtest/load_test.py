"""Real load test client for the inference service.

Generates real concurrent HTTP traffic against a RUNNING instance of the
service (default http://127.0.0.1:8000) using httpx.AsyncClient, measures
real wall-clock latency per request, and reports measured P50/P95/P99
latency and throughput. No numbers here are estimated — they come from
actually timing real HTTP round trips.

Usage:
    python loadtest/load_test.py --concurrency 32 --requests 2000
    python loadtest/load_test.py --concurrency 8 --requests 500 --cache-friendly

By default each request carries a different random input (worst case for
the cache: every request is a forward pass, bounded only by batching). Pass
--cache-friendly to reuse a small pool of inputs and exercise the cache hit
path instead.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import time
from dataclasses import dataclass, field

import httpx

DEFAULT_INPUT_DIM = 3 * 32 * 32


@dataclass
class RequestResult:
    latency_ms: float
    status_code: int
    cached: bool = False
    error: str | None = None


@dataclass
class LoadTestReport:
    total_requests: int
    duration_s: float
    results: list[RequestResult] = field(default_factory=list)

    @property
    def successes(self) -> list[RequestResult]:
        return [r for r in self.results if r.status_code == 200]

    @property
    def errors(self) -> list[RequestResult]:
        return [r for r in self.results if r.status_code != 200]

    def percentile(self, p: float) -> float:
        latencies = sorted(r.latency_ms for r in self.successes)
        if not latencies:
            return float("nan")
        k = (len(latencies) - 1) * (p / 100)
        f = int(k)
        c = min(f + 1, len(latencies) - 1)
        if f == c:
            return latencies[f]
        return latencies[f] + (latencies[c] - latencies[f]) * (k - f)

    def throughput_rps(self) -> float:
        return self.total_requests / self.duration_s if self.duration_s > 0 else float("nan")

    def cache_hit_rate(self) -> float:
        successes = self.successes
        if not successes:
            return float("nan")
        hits = sum(1 for r in successes if r.cached)
        return hits / len(successes)

    def summary(self) -> str:
        lines = [
            "=== Load Test Report (measured, not estimated) ===",
            f"Total requests:      {self.total_requests}",
            f"Successful:          {len(self.successes)}",
            f"Errors (non-200):    {len(self.errors)}",
            f"Duration:            {self.duration_s:.3f} s",
            f"Throughput:          {self.throughput_rps():.2f} req/s",
            f"Cache hit rate:      {self.cache_hit_rate() * 100:.1f}%",
            f"Latency P50:         {self.percentile(50):.2f} ms",
            f"Latency P95:         {self.percentile(95):.2f} ms",
            f"Latency P99:         {self.percentile(99):.2f} ms",
            f"Latency min/max:     "
            f"{min((r.latency_ms for r in self.successes), default=float('nan')):.2f} / "
            f"{max((r.latency_ms for r in self.successes), default=float('nan')):.2f} ms",
        ]
        return "\n".join(lines)


def _random_input(dim: int) -> list[float]:
    return [random.random() for _ in range(dim)]


async def _one_request(
    client: httpx.AsyncClient, values: list[float]
) -> RequestResult:
    start = time.perf_counter()
    try:
        resp = await client.post("/infer", json={"input": values})
        latency_ms = (time.perf_counter() - start) * 1000
        cached = False
        if resp.status_code == 200:
            cached = bool(resp.json().get("cached", False))
        return RequestResult(latency_ms=latency_ms, status_code=resp.status_code, cached=cached)
    except httpx.HTTPError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return RequestResult(latency_ms=latency_ms, status_code=-1, error=str(exc))


async def run_load_test(
    base_url: str,
    total_requests: int,
    concurrency: int,
    input_dim: int,
    cache_friendly: bool,
) -> LoadTestReport:
    if cache_friendly:
        pool = [_random_input(input_dim) for _ in range(max(1, concurrency // 4))]
        inputs = [random.choice(pool) for _ in range(total_requests)]
    else:
        inputs = [_random_input(input_dim) for _ in range(total_requests)]

    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        async def bound_request(values: list[float]) -> None:
            async with semaphore:
                result = await _one_request(client, values)
                results.append(result)

        start = time.perf_counter()
        await asyncio.gather(*(bound_request(v) for v in inputs))
        duration_s = time.perf_counter() - start

    return LoadTestReport(total_requests=total_requests, duration_s=duration_s, results=results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test the inference service")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=DEFAULT_INPUT_DIM)
    parser.add_argument(
        "--cache-friendly",
        action="store_true",
        help="Reuse a small pool of inputs to exercise cache hits instead of worst-case misses",
    )
    args = parser.parse_args()

    report = asyncio.run(
        run_load_test(
            base_url=args.base_url,
            total_requests=args.requests,
            concurrency=args.concurrency,
            input_dim=args.input_dim,
            cache_friendly=args.cache_friendly,
        )
    )
    print(report.summary())

    if report.errors:
        sample = report.errors[0]
        print(f"\nSample error: status={sample.status_code} detail={sample.error}")


if __name__ == "__main__":
    main()
