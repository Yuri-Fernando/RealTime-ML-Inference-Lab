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

Concurrency sweep (capacity planning):
    python loadtest/load_test.py --sweep 1,2,4,8,16,32,64 --requests-per-level 300

Runs the SAME measured load test at each concurrency level in sequence
against the same running service instance, and writes a single
machine-generated report (JSON + Markdown) to loadtest/reports/ with
P50/P95/P99/throughput per level and the saturation point actually observed
(first level where P95 latency grows disproportionately and/or error rate
rises) — not a guessed number. This is real per-hardware capacity planning
for THIS model/service/GPU, not a generic claim.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

DEFAULT_INPUT_DIM = 3 * 32 * 32


@dataclass
class RequestResult:
    latency_ms: float
    status_code: int
    cached: bool = False
    error: str | None = None
    # Cenário LLM (V0.3): preenchidos a partir da resposta de /generate.
    completion_tokens: int | None = None
    ttft_ms: float | None = None


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

    def tokens_per_second(self) -> float | None:
        """Tokens gerados por segundo de parede (só cenário LLM)."""
        toks = [r.completion_tokens for r in self.successes if r.completion_tokens is not None]
        if not toks or self.duration_s <= 0:
            return None
        return sum(toks) / self.duration_s

    def ttft_percentile(self, p: float) -> float | None:
        vals = sorted(r.ttft_ms for r in self.successes if r.ttft_ms is not None)
        if not vals:
            return None
        k = (len(vals) - 1) * (p / 100)
        f = int(k)
        c = min(f + 1, len(vals) - 1)
        return vals[f] if f == c else vals[f] + (vals[c] - vals[f]) * (k - f)

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


# Prompts curtos e variados do cenário LLM (mesmo espírito do quality_set do serviço).
LLM_PROMPTS = (
    "Explique em uma frase o que é batching dinâmico.",
    "What is 2 + 2? Answer with just the number.",
    "Liste três cores primárias.",
    "Translate to English: 'O gato dorme no sofá.'",
    "Qual é a capital do Brasil?",
    "Write a one-line Python function that returns the square of x.",
    "Resuma em uma frase: servir modelos com baixa latência exige fila, lote e métricas.",
    "Name two benefits of quantizing a language model.",
    "Dê um exemplo de contrapressão em um sistema de filas.",
    "Why does the p99 latency matter more than the mean?",
)


async def _one_llm_request(
    client: httpx.AsyncClient, prompt: str, max_new_tokens: int
) -> RequestResult:
    start = time.perf_counter()
    try:
        resp = await client.post(
            "/generate", json={"prompt": prompt, "max_new_tokens": max_new_tokens}
        )
        latency_ms = (time.perf_counter() - start) * 1000
        tokens = ttft = None
        if resp.status_code == 200:
            body = resp.json()
            tokens = body.get("completion_tokens")
            ttft = body.get("ttft_ms")
        return RequestResult(
            latency_ms=latency_ms,
            status_code=resp.status_code,
            completion_tokens=tokens,
            ttft_ms=ttft,
        )
    except httpx.HTTPError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return RequestResult(latency_ms=latency_ms, status_code=-1, error=str(exc))


async def run_load_test(
    base_url: str,
    total_requests: int,
    concurrency: int,
    input_dim: int,
    cache_friendly: bool,
    scenario: str = "cnn",
    max_new_tokens: int = 32,
) -> LoadTestReport:
    if scenario == "llm":
        return await _run_llm_load_test(base_url, total_requests, concurrency, max_new_tokens)
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


async def _run_llm_load_test(
    base_url: str, total_requests: int, concurrency: int, max_new_tokens: int
) -> LoadTestReport:
    prompts = [random.choice(LLM_PROMPTS) for _ in range(total_requests)]
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    # Geração leva segundos: timeout bem maior que o do cenário CNN.
    async with httpx.AsyncClient(base_url=base_url, timeout=600.0) as client:
        async def bound(prompt: str) -> None:
            async with semaphore:
                results.append(await _one_llm_request(client, prompt, max_new_tokens))

        start = time.perf_counter()
        await asyncio.gather(*(bound(p) for p in prompts))
        duration_s = time.perf_counter() - start

    return LoadTestReport(total_requests=total_requests, duration_s=duration_s, results=results)


@dataclass
class SweepLevelResult:
    concurrency: int
    total_requests: int
    duration_s: float
    throughput_rps: float
    error_rate: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    # Cenário LLM (None no cenário CNN)
    tokens_per_s: float | None = None
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None

    @classmethod
    def from_report(cls, concurrency: int, report: LoadTestReport) -> "SweepLevelResult":
        error_rate = (
            len(report.errors) / report.total_requests if report.total_requests else float("nan")
        )
        return cls(
            concurrency=concurrency,
            total_requests=report.total_requests,
            duration_s=report.duration_s,
            throughput_rps=report.throughput_rps(),
            error_rate=error_rate,
            p50_ms=report.percentile(50),
            p95_ms=report.percentile(95),
            p99_ms=report.percentile(99),
            tokens_per_s=report.tokens_per_second(),
            ttft_p50_ms=report.ttft_percentile(50),
            ttft_p95_ms=report.ttft_percentile(95),
        )


def _find_saturation_point(levels: list[SweepLevelResult]) -> "int | None":
    """Identify the first concurrency level where the system shows real
    saturation, using two simple, measured (not guessed) signals:

      1. error_rate > 1% at that level (backpressure/429s/timeouts kicking
         in for real), OR
      2. P95 latency more than DOUBLES relative to the previous level while
         throughput stops growing proportionally (classic queueing-delay
         knee: latency balloons, throughput plateaus).

    Returns the concurrency value of the first level matching either
    condition, or None if no level in the sweep saturated the service
    (i.e. the whole tested range stayed within the stable region).
    """
    for i, level in enumerate(levels):
        if level.error_rate > 0.01:
            return level.concurrency
        if i > 0:
            prev = levels[i - 1]
            if prev.p95_ms > 0 and level.p95_ms > prev.p95_ms * 2:
                throughput_growth = (
                    (level.throughput_rps - prev.throughput_rps) / prev.throughput_rps
                    if prev.throughput_rps > 0
                    else float("inf")
                )
                if throughput_growth < 0.15:  # throughput basically stopped scaling
                    return level.concurrency
    return None


async def run_concurrency_sweep(
    base_url: str,
    levels: list[int],
    requests_per_level: int,
    input_dim: int,
    cache_friendly: bool,
    scenario: str = "cnn",
    max_new_tokens: int = 32,
) -> list[SweepLevelResult]:
    """Run the same real load test at each concurrency level, in sequence,
    against the same running service instance. Sequential (not parallel)
    sweeping is deliberate: each level needs a clean measurement window,
    and running levels concurrently would conflate their load."""
    results: list[SweepLevelResult] = []
    for concurrency in levels:
        report = await run_load_test(
            base_url=base_url,
            total_requests=requests_per_level,
            concurrency=concurrency,
            input_dim=input_dim,
            cache_friendly=cache_friendly,
            scenario=scenario,
            max_new_tokens=max_new_tokens,
        )
        results.append(SweepLevelResult.from_report(concurrency, report))
        # Brief pause between levels so one level's tail requests don't
        # bleed into the next level's measurement window.
        await asyncio.sleep(0.5)
    return results


def _write_sweep_report(
    levels: list[SweepLevelResult],
    saturation_point: "int | None",
    output_dir: Path,
    input_dim: int,
    requests_per_level: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    json_path = output_dir / f"capacity-sweep-{timestamp}.json"
    json_payload = {
        "timestamp": timestamp,
        "input_dim": input_dim,
        "requests_per_level": requests_per_level,
        "saturation_point_concurrency": saturation_point,
        "levels": [
            {
                "concurrency": lvl.concurrency,
                "total_requests": lvl.total_requests,
                "duration_s": lvl.duration_s,
                "throughput_rps": lvl.throughput_rps,
                "error_rate": lvl.error_rate,
                "p50_ms": lvl.p50_ms,
                "p95_ms": lvl.p95_ms,
                "p99_ms": lvl.p99_ms,
                **(
                    {
                        "tokens_per_s": lvl.tokens_per_s,
                        "ttft_p50_ms": lvl.ttft_p50_ms,
                        "ttft_p95_ms": lvl.ttft_p95_ms,
                    }
                    if lvl.tokens_per_s is not None
                    else {}
                ),
            }
            for lvl in levels
        ],
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    md_path = output_dir / f"capacity-sweep-{timestamp}.md"
    md_lines = [
        f"# Capacity sweep report — {timestamp}",
        "",
        "Real measured load test run at each concurrency level in sequence "
        "against this machine's GPU/CPU and this model. Not estimated, not "
        "extrapolated — every row below comes from an actual run of "
        "`loadtest/load_test.py`.",
        "",
        f"- Requests per level: {requests_per_level}",
        f"- Input dim: {input_dim}",
        "",
    ]
    is_llm = any(lvl.tokens_per_s is not None for lvl in levels)
    if is_llm:
        md_lines += [
            "| Concurrency | Requests | Duration (s) | Throughput (req/s) | Tokens/s | Error rate | "
            "P50 (ms) | P95 (ms) | P99 (ms) | TTFT P50 (ms) | TTFT P95 (ms) |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    else:
        md_lines += [
            "| Concurrency | Requests | Duration (s) | Throughput (req/s) | Error rate | P50 (ms) | P95 (ms) | P99 (ms) |",
            "|---|---|---|---|---|---|---|---|",
        ]
    for lvl in levels:
        if is_llm:
            ttft50 = f"{lvl.ttft_p50_ms:.0f}" if lvl.ttft_p50_ms is not None else "n/d"
            ttft95 = f"{lvl.ttft_p95_ms:.0f}" if lvl.ttft_p95_ms is not None else "n/d"
            tps = f"{lvl.tokens_per_s:.1f}" if lvl.tokens_per_s is not None else "n/d"
            md_lines.append(
                f"| {lvl.concurrency} | {lvl.total_requests} | {lvl.duration_s:.2f} | "
                f"{lvl.throughput_rps:.2f} | {tps} | {lvl.error_rate * 100:.1f}% | "
                f"{lvl.p50_ms:.0f} | {lvl.p95_ms:.0f} | {lvl.p99_ms:.0f} | {ttft50} | {ttft95} |"
            )
        else:
            md_lines.append(
                f"| {lvl.concurrency} | {lvl.total_requests} | {lvl.duration_s:.2f} | "
                f"{lvl.throughput_rps:.2f} | {lvl.error_rate * 100:.1f}% | "
                f"{lvl.p50_ms:.1f} | {lvl.p95_ms:.1f} | {lvl.p99_ms:.1f} |"
            )
    md_lines.append("")
    if saturation_point is not None:
        md_lines.append(
            f"**Saturation point (measured): concurrency = {saturation_point}.** "
            "This is the first tested level where either the error rate exceeded "
            "1% or P95 latency more than doubled relative to the previous level "
            "while throughput stopped growing — i.e. the queueing-delay knee was "
            "observed directly in this run, not inferred from a formula."
        )
    else:
        md_lines.append(
            "**No saturation point observed within the tested concurrency range.** "
            "The service stayed in its stable region for every level tested here; "
            "a real saturation point may exist at a higher, untested concurrency."
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


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
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Comma-separated concurrency levels to run a capacity sweep, "
        "e.g. --sweep 1,2,4,8,16,32,64",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=300,
        help="Requests to send at each concurrency level during a --sweep run",
    )
    parser.add_argument(
        "--scenario",
        choices=["cnn", "llm"],
        default="cnn",
        help="cnn = POST /infer (TinyCNN, padrão); llm = POST /generate (inference_service.llm_app)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Tokens a gerar por requisição no cenário llm",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=str(Path(__file__).parent / "reports"),
        help="Directory to write capacity-sweep JSON/Markdown reports into",
    )
    args = parser.parse_args()

    if args.sweep:
        levels = [int(x.strip()) for x in args.sweep.split(",") if x.strip()]
        print(f"=== Concurrency sweep (measured, not estimated): {levels} ===")
        sweep_results = asyncio.run(
            run_concurrency_sweep(
                base_url=args.base_url,
                levels=levels,
                requests_per_level=args.requests_per_level,
                input_dim=args.input_dim,
                cache_friendly=args.cache_friendly,
                scenario=args.scenario,
                max_new_tokens=args.max_new_tokens,
            )
        )
        for lvl in sweep_results:
            print(
                f"concurrency={lvl.concurrency:>4}  "
                f"throughput={lvl.throughput_rps:7.2f} req/s  "
                f"p50={lvl.p50_ms:8.2f} ms  p95={lvl.p95_ms:8.2f} ms  p99={lvl.p99_ms:8.2f} ms  "
                f"errors={lvl.error_rate * 100:5.1f}%"
            )
        saturation = _find_saturation_point(sweep_results)
        if saturation is not None:
            print(f"\nMeasured saturation point: concurrency = {saturation}")
        else:
            print("\nNo saturation point observed within the tested range.")

        json_path, md_path = _write_sweep_report(
            sweep_results,
            saturation,
            Path(args.report_dir),
            input_dim=args.input_dim,
            requests_per_level=args.requests_per_level,
        )
        print(f"\nReport written to:\n  {json_path}\n  {md_path}")
        return

    report = asyncio.run(
        run_load_test(
            base_url=args.base_url,
            total_requests=args.requests,
            concurrency=args.concurrency,
            input_dim=args.input_dim,
            cache_friendly=args.cache_friendly,
            scenario=args.scenario,
            max_new_tokens=args.max_new_tokens,
        )
    )
    print(report.summary())
    tps = report.tokens_per_second()
    if tps is not None:
        print(f"Tokens/s:            {tps:.1f}")
        print(f"TTFT P50/P95:        {report.ttft_percentile(50)} / {report.ttft_percentile(95)} ms")

    if report.errors:
        sample = report.errors[0]
        print(f"\nSample error: status={sample.status_code} detail={sample.error}")


if __name__ == "__main__":
    main()
