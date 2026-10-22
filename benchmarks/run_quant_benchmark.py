"""Benchmark de quantização do HuggingFaceBackend (V0.3) — tudo medido, nada estimado.

Para cada modo (fp32, fp16, bf16, int8, nf4) este script:

  1. sobe o serviço REAL (`uvicorn inference_service.llm_app:app`) num
     subprocesso, com RTML_BACKEND=hf e RTML_HF_QUANT=<modo>;
  2. espera o /health responder e coleta /quality (perplexidade + saídas greedy);
  3. aquece o modelo e roda o MESMO load test do repositório
     (`loadtest/load_test.py --scenario llm`, concurrency sweep) contra o serviço;
  4. amostra `nvidia-smi` (uso de GPU e memória usada) durante o sweep;
  5. lê o /health (peso do modelo, pico de VRAM alocada pelo PyTorch);
  6. derruba o servidor e passa ao próximo modo.

No fim grava `benchmarks/results/quant-<timestamp>.{json,md}` e gráficos PNG.

Uso (do diretório raiz do repositório, com o Python 3.10 canônico):
    python benchmarks/run_quant_benchmark.py --modes fp32,fp16,bf16,int8,nf4 \
        --sweep 1,2,4,8,16 --requests-per-level 24 --max-new-tokens 32

Notas de honestidade (também gravadas no JSON):
- A GPU é compartilhada com outros processos da máquina; por isso o baseline
  de `nvidia-smi` antes do servidor é registrado e subtraído da coluna de
  "VRAM do processo (nvidia-smi)".
- O mesmo modelo, os mesmos prompts e o mesmo `max_new_tokens` valem para
  todos os modos; os prompts são sorteados aleatoriamente por requisição.
- Qualidade: perplexidade e concordância greedy contra a referência (fp32) são
  sinais baratos de degradação, NÃO um benchmark de tarefa.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loadtest"))
import load_test  # noqa: E402  (módulo standalone do repositório)

DEFAULT_MODES = "fp32,fp16,bf16,int8,nf4"


# ----------------------------------------------------------------- nvidia-smi
def _smi() -> tuple[float, float] | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip().splitlines()[0]
        util, mem = (float(x) for x in out.split(","))
        return util, mem
    except Exception:
        return None


def _probe_idle(seconds: float = 4.0) -> dict | None:
    """Média de uso e memória da GPU SEM o servidor de teste rodando.

    A GPU é compartilhada com outros processos da máquina; este baseline por
    modo permite subtrair a memória deles e sinalizar medições contaminadas
    (uso ocioso alto = outro processo disputando a GPU durante o teste).
    """
    pts = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        s = _smi()
        if s is not None:
            pts.append(s)
        time.sleep(0.5)
    if not pts:
        return None
    return {
        "util_mean_pct": round(sum(u for u, _ in pts) / len(pts), 1),
        "mem_used_mib": round(sum(m for _, m in pts) / len(pts)),
    }


def _wait_for_idle_gpu(max_wait_s: float, util_max: float = 20.0, mem_max_mib: float = 3000.0) -> dict:
    """Espera a GPU compartilhada ficar ociosa (outros processos da máquina).

    Latência e vazão só são comparáveis entre modos se ninguém mais estiver
    usando a GPU. Se não ficar ociosa dentro de `max_wait_s`, segue assim
    mesmo e marca `clean=False` — a medição fica sinalizada, não escondida.
    """
    t0 = time.perf_counter()
    while True:
        idle = _probe_idle(4.0) or {"util_mean_pct": None, "mem_used_mib": None}
        ok = (
            idle["util_mean_pct"] is not None
            and idle["util_mean_pct"] <= util_max
            and idle["mem_used_mib"] <= mem_max_mib
        )
        waited = round(time.perf_counter() - t0, 1)
        if ok or waited >= max_wait_s:
            return {**idle, "waited_s": waited, "clean": ok}
        time.sleep(15)


class GpuSampler:
    """Amostra utilização e memória da GPU a cada `period` segundos."""

    def __init__(self, period: float = 0.5) -> None:
        self.period = period
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            s = _smi()
            if s is not None:
                self.samples.append(s)
            self._stop.wait(self.period)

    def __enter__(self) -> "GpuSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self) -> dict:
        if not self.samples:
            return {"samples": 0}
        utils = [u for u, _ in self.samples]
        mems = [m for _, m in self.samples]
        return {
            "samples": len(self.samples),
            "util_mean_pct": round(sum(utils) / len(utils), 1),
            "util_max_pct": max(utils),
            "mem_used_max_mib": max(mems),
        }


# --------------------------------------------------------------------- server
def _start_server(mode: str, port: int, log_path: Path, args) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(
        {
            "RTML_BACKEND": "hf",
            "RTML_HF_QUANT": mode,
            "RTML_HF_MODEL": args.model,
            "RTML_HF_LOCAL_ONLY": "1",
            "HF_HUB_OFFLINE": "1",
            "RTML_LLM_MAX_BATCH_SIZE": str(args.max_batch_size),
            "RTML_LLM_MAX_BATCH_WAIT_MS": str(args.max_wait_ms),
            "RTML_LLM_MAX_QUEUE_DEPTH": "256",
            "RTML_LLM_REQUEST_TIMEOUT_S": "600",
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",  # se o servidor morrer por crash nativo, o log mostra onde
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "inference_service.llm_app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def _wait_ready(base: str, proc: subprocess.Popen, timeout_s: float) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"servidor encerrou com código {proc.returncode} antes de ficar pronto")
        try:
            r = httpx.get(f"{base}/health", timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return time.perf_counter() - t0
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"servidor não ficou pronto em {timeout_s}s")


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


# ------------------------------------------------------------------- one mode
def _run_mode_once(mode: str, args, results_dir: Path, log_dir: Path) -> dict:
    base = f"http://127.0.0.1:{args.port}"
    idle = _wait_for_idle_gpu(args.idle_wait_s)
    baseline = (idle["util_mean_pct"], idle["mem_used_mib"]) if idle["mem_used_mib"] is not None else None
    out: dict = {"mode": mode, "status": "ok", "gpu_baseline": baseline, "gpu_idle": idle}
    proc = _start_server(mode, args.port, log_dir / f"server-{mode}.log", args)
    try:
        try:
            out["startup_seconds"] = round(_wait_ready(base, proc, args.startup_timeout), 1)
        except Exception as exc:  # modo que não carrega: registrar o limite real
            out.update(status="failed_to_start", error=str(exc))
            return out

        out["quality"] = httpx.get(f"{base}/quality", timeout=300).json()

        # aquecimento (kernels CUDA / autotune) fora da medição
        for p in load_test.LLM_PROMPTS[:3]:
            httpx.post(
                f"{base}/generate",
                json={"prompt": p, "max_new_tokens": args.max_new_tokens},
                timeout=300,
            )

        levels = [int(x) for x in args.sweep.split(",")]
        with GpuSampler() as sampler:
            sweep = asyncio.run(
                load_test.run_concurrency_sweep(
                    base_url=base,
                    levels=levels,
                    requests_per_level=args.requests_per_level,
                    input_dim=0,
                    cache_friendly=False,
                    scenario="llm",
                    max_new_tokens=args.max_new_tokens,
                )
            )
        out["gpu_sampler"] = sampler.summary()
        out["health"] = httpx.get(f"{base}/health", timeout=30).json()
        # Sinais de contaminação por outros processos na mesma GPU:
        #  - não estava ociosa antes de começar;
        #  - a memória usada na GPU durante o sweep passou muito do que este
        #    processo alocou (pico PyTorch + contexto CUDA + folga).
        peak = out["health"].get("gpu_peak_allocated_bytes", 0) / 2**20
        used_by_proc = (out["gpu_sampler"].get("mem_used_max_mib") or 0) - (idle["mem_used_mib"] or 0)
        out["mem_anomaly"] = bool(used_by_proc > peak + 2000 or used_by_proc < 0)
        out["measurement_clean"] = bool(idle["clean"] and not out["mem_anomaly"])
        out["sweep"] = [asdict(lvl) for lvl in sweep]
        out["saturation_point_concurrency"] = load_test._find_saturation_point(sweep)
        return out
    finally:
        _stop_server(proc)
        time.sleep(4)  # deixa o driver liberar a VRAM antes do próximo modo


def run_mode(mode: str, args, results_dir: Path, log_dir: Path) -> dict:
    """Executa um modo com até `--attempts` tentativas.

    Se o servidor morrer ou perder a conexão no meio (ex.: crash nativo do
    bitsandbytes sob disputa de GPU com outros processos da máquina), o erro e a
    cauda do log do servidor ficam registrados em `attempt_errors` — a falha não
    é escondida, e a tentativa seguinte começa de um servidor limpo.
    """
    errors: list[dict] = []
    for attempt in range(1, args.attempts + 1):
        try:
            out = _run_mode_once(mode, args, results_dir, log_dir)
            if out["status"] == "ok" or attempt == args.attempts:
                out["attempts"] = attempt
                if errors:
                    out["attempt_errors"] = errors
                return out
            errors.append({"attempt": attempt, "error": out.get("error")})
        except (httpx.HTTPError, RuntimeError, TimeoutError) as exc:
            log = log_dir / f"server-{mode}.log"
            tail = ""
            try:
                raw = log.read_text(encoding="utf-8", errors="replace")
                lines = [ln for ln in raw.replace(chr(13), chr(10)).splitlines() if "Loading weights" not in ln]
                tail = chr(10).join(lines)[-1500:]
            except OSError:
                pass
            errors.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}", "server_log_tail": tail})
            print(f"{mode}: tentativa {attempt} falhou ({type(exc).__name__}); "
                  f"{'tentando de novo' if attempt < args.attempts else 'desistindo'}", flush=True)
            time.sleep(8)
    return {"mode": mode, "status": "failed_during_run", "attempts": args.attempts,
            "attempt_errors": errors, "error": errors[-1]["error"], "gpu_baseline": None}


# --------------------------------------------------------------------- report
def _agreement(ref: list[str], got: list[str]) -> dict:
    ratios = [difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(ref, got)]
    return {
        "exact_match": sum(1 for a, b in zip(ref, got) if a == b),
        "of": len(ref),
        "mean_similarity": round(sum(ratios) / len(ratios), 4) if ratios else None,
    }


def write_report(runs: list[dict], meta: dict, out_dir: Path, stamp: str) -> tuple[Path, Path]:
    ref_mode = meta["quality_reference"]
    ref_run = next((r for r in runs if r["mode"] == ref_mode and r["status"] == "ok"), None)
    for r in runs:
        if r["status"] != "ok":
            continue
        q = r["quality"]
        r["quality_vs_reference"] = (
            _agreement(ref_run["quality"]["greedy_outputs"], q["greedy_outputs"]) if ref_run else None
        )

    payload = {"meta": meta, "runs": runs}
    json_path = out_dir / f"quant-{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    ok = [r for r in runs if r["status"] == "ok"]
    top = meta["report_concurrency"]

    def at(r: dict, c: int) -> dict | None:
        return next((lvl for lvl in r["sweep"] if lvl["concurrency"] == c), None)

    md = [
        f"# Quantização no HuggingFaceBackend — {stamp}",
        "",
        f"Modelo: `{meta['model']}` · GPU: {meta['gpu']} · torch {meta['torch']} · "
        f"transformers {meta['transformers']} · bitsandbytes {meta['bitsandbytes']}",
        f"Commit do repositório na execução: `{meta['git_head']}` "
        f"(working tree {'com' if meta['git_dirty'] else 'sem'} alterações não commitadas).",
        f"Carga: {meta['requests_per_level']} requisições por nível de concorrência, "
        f"{meta['max_new_tokens']} tokens novos, lote máx. {meta['max_batch_size']}, "
        f"janela {meta['max_wait_ms']} ms, decodificação greedy, sem cache de resultado.",
        "",
        "Tudo abaixo foi medido nesta execução (nada estimado). A GPU é compartilhada com outros "
        f"processos da máquina; baseline antes do servidor: {meta['gpu_baseline_mib']} MiB usados.",
        "",
        f"## Resumo (concorrência {top})",
        "",
        "| Modo | Pesos (MiB) | Pico VRAM PyTorch (MiB) | VRAM processo (MiB, nvidia-smi) | "
        "P50 (ms) | P95 (ms) | P99 (ms) | req/s | tokens/s | TTFT P50 (ms) | GPU util média (%) | "
        "Perplexidade | Greedy = ref. | Similaridade | GPU ociosa antes (util %, MiB) | Medição limpa? |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        if r["status"] != "ok":
            md.append(f"| {r['mode']} | — | — | — | — | — | — | — | — | — | — | — | — | "
                      f"**{r['status']}**: {r.get('error', '')[:80]} |")
            continue
        h = r["health"]
        lvl = at(r, top) or r["sweep"][-1]
        base_mem = (r["gpu_baseline"] or (0, 0))[1]
        proc_mem = (r["gpu_sampler"].get("mem_used_max_mib", 0) or 0) - base_mem
        qa = r["quality_vs_reference"]
        qa_exact = f"{qa['exact_match']}/{qa['of']}" if qa else "n/d (sem referência)"
        qa_sim = qa["mean_similarity"] if qa else "n/d"
        md.append(
            f"| {r['mode']} | {h['memory_footprint_bytes'] / 2**20:.0f} | "
            f"{h.get('gpu_peak_allocated_bytes', 0) / 2**20:.0f} | {proc_mem:.0f} | "
            f"{lvl['p50_ms']:.0f} | {lvl['p95_ms']:.0f} | {lvl['p99_ms']:.0f} | "
            f"{lvl['throughput_rps']:.2f} | {lvl['tokens_per_s']:.1f} | "
            f"{(lvl['ttft_p50_ms'] or float('nan')):.0f} | "
            f"{r['gpu_sampler'].get('util_mean_pct', float('nan'))} | "
            f"{r['quality']['perplexity']:.2f} | "
            f"{qa_exact} | {qa_sim} | "
            f"{(r.get('gpu_idle') or {}).get('util_mean_pct', 'n/d')}%, "
            f"{(r.get('gpu_idle') or {}).get('mem_used_mib', 'n/d')} | "
            f"{'sim' if r.get('measurement_clean') else '**NÃO — GPU disputada, não comparar latência**'} |"
        )
    md += ["", "## Concurrency sweep por modo", ""]
    for r in ok:
        md += [
            f"### {r['mode']}  (sobe em {r['startup_seconds']} s; saturação medida: "
            f"{r['saturation_point_concurrency']})",
            "",
            "| Conc. | Req. | Duração (s) | req/s | tokens/s | Erros | P50 | P95 | P99 | TTFT P50 | TTFT P95 |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for lvl in r["sweep"]:
            md.append(
                f"| {lvl['concurrency']} | {lvl['total_requests']} | {lvl['duration_s']:.1f} | "
                f"{lvl['throughput_rps']:.2f} | {lvl['tokens_per_s']:.1f} | "
                f"{lvl['error_rate'] * 100:.1f}% | {lvl['p50_ms']:.0f} | {lvl['p95_ms']:.0f} | "
                f"{lvl['p99_ms']:.0f} | {(lvl['ttft_p50_ms'] or float('nan')):.0f} | "
                f"{(lvl['ttft_p95_ms'] or float('nan')):.0f} |"
            )
        md.append("")
    md += [
        "## Limites desta medição",
        "",
        "- Um único modelo pequeno (0,5 B parâmetros) numa única GPU de 8 GB compartilhada; "
        "os resultados não se extrapolam para modelos maiores nem para outras GPUs.",
        "- INT8 (LLM.int8() do bitsandbytes) costuma ser MAIS LENTO que FP16 em modelos pequenos: "
        "ele economiza memória, não tempo. NF4 também tem custo de dequantização.",
        "- Perplexidade e concordância greedy sobre 6 textos / 8 prompts fixos são um sinal "
        "barato de degradação, não um benchmark de qualidade de tarefa.",
        "- O TTFT é o tempo de espera em fila + prefill do lote (medido no backend), não "
        "streaming token a token.",
    ]
    md_path = out_dir / f"quant-{stamp}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    _charts(ok, out_dir, stamp)
    return json_path, md_path


def _charts(ok: list[dict], out_dir: Path, stamp: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not ok:
        return
    modes = [r["mode"] for r in ok]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(modes, [r["health"]["memory_footprint_bytes"] / 2**20 for r in ok], color="#4c78a8")
    ax[0].set_title("Memória dos pesos (MiB)")
    ax[1].bar(modes, [r["health"].get("gpu_peak_allocated_bytes", 0) / 2**20 for r in ok], color="#f58518")
    ax[1].set_title("Pico de VRAM alocada pelo PyTorch (MiB)")
    ax[2].bar(modes, [r["quality"]["perplexity"] for r in ok], color="#54a24b")
    ax[2].set_title("Perplexidade (menor = melhor)")
    for a in ax:
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"quant-{stamp}-memory-quality.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for r in ok:
        xs = [lvl["concurrency"] for lvl in r["sweep"]]
        ax[0].plot(xs, [lvl["p50_ms"] for lvl in r["sweep"]], marker="o", label=r["mode"])
        ax[1].plot(xs, [lvl["p95_ms"] for lvl in r["sweep"]], marker="o", label=r["mode"])
        ax[2].plot(xs, [lvl["tokens_per_s"] for lvl in r["sweep"]], marker="o", label=r["mode"])
    for a, t in zip(ax, ["Latência P50 (ms)", "Latência P95 (ms)", "Tokens/s"]):
        a.set_title(t)
        a.set_xlabel("concorrência")
        a.set_xscale("log", base=2)
        a.grid(alpha=0.3)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"quant-{stamp}-latency-throughput.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------- main
def _versions() -> dict:
    import torch
    import transformers

    try:
        import bitsandbytes

        bnb = bitsandbytes.__version__
    except Exception:
        bnb = "ausente"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sem GPU"
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "bitsandbytes": bnb, "gpu": gpu}


def _git(root: Path) -> tuple[str, bool]:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                              capture_output=True, text=True).stdout.strip() or "n/d"
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                    capture_output=True, text=True).stdout.strip())
        return head, dirty
    except Exception:
        return "n/d", False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modes", default=DEFAULT_MODES)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--sweep", default="1,2,4,8,16")
    ap.add_argument("--requests-per-level", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--max-wait-ms", type=float, default=20.0)
    ap.add_argument("--report-concurrency", type=int, default=8)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--startup-timeout", type=float, default=300.0)
    ap.add_argument("--quality-reference", default="fp32")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    ap.add_argument("--idle-wait-s", type=float, default=600.0,
                    help="Espera máxima (por modo) até a GPU compartilhada ficar ociosa")
    ap.add_argument("--attempts", type=int, default=2,
                    help="Tentativas por modo se o servidor morrer/perder a conexão no meio")
    ap.add_argument("--resume", default=None,
                    help="Arquivo quant-*.partial.json de uma execução interrompida: reaproveita os "
                         "modos já concluídos (status ok) e roda só os que faltam")
    ap.add_argument("--git-root", default=str(ROOT),
                    help="Repositório git de onde ler o commit (útil ao rodar numa cópia em staging)")
    ap.add_argument("--log-dir", default=r"D:\Claude Temp\staging\realtime-v03\logs")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    head, dirty = _git(Path(args.git_root))
    base = _smi()
    meta = {
        "timestamp": stamp,
        "model": args.model,
        "sweep": args.sweep,
        "requests_per_level": args.requests_per_level,
        "max_new_tokens": args.max_new_tokens,
        "max_batch_size": args.max_batch_size,
        "max_wait_ms": args.max_wait_ms,
        "report_concurrency": args.report_concurrency,
        "quality_reference": args.quality_reference,
        "git_head": head,
        "git_dirty": dirty,
        "gpu_baseline_mib": base[1] if base else None,
        **_versions(),
    }

    runs: list[dict] = []
    if args.resume:
        prev = json.loads(Path(args.resume).read_text(encoding="utf-8"))
        meta = prev["meta"]
        stamp = meta["timestamp"]
        meta["resumed"] = True
        runs = [r for r in prev["runs"] if r["status"] == "ok"]
        print(f"retomando {stamp}: já concluídos {[r['mode'] for r in runs]}", flush=True)
    done = {r["mode"] for r in runs}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip() and m.strip() not in done]:
        print(f"\n=== {mode} ===", flush=True)
        run = run_mode(mode, args, out_dir, Path(args.log_dir))
        print(f"{mode}: {run['status']}", flush=True)
        runs.append(run)
        # grava parcial a cada modo: nada se perde se uma execução longa for interrompida
        (out_dir / f"quant-{stamp}.partial.json").write_text(
            json.dumps({"meta": meta, "runs": runs}, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    json_path, md_path = write_report(runs, meta, out_dir, stamp)
    (out_dir / f"quant-{stamp}.partial.json").unlink(missing_ok=True)
    print(f"\nRelatório:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
