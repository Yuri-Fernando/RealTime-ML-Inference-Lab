"""HuggingFaceBackend com o MODELO REAL (Qwen2.5-0.5B-Instruct, ~1-2 GB).

Estes testes carregam o modelo de verdade. São pulados (com motivo explícito,
nunca silenciosamente aprovados) se não houver GPU CUDA ou se o modelo não
estiver no cache local do Hugging Face (nada é baixado pelos testes).
"""
import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from inference_service.backends.base import GenerationRequest  # noqa: E402
from inference_service.config import settings  # noqa: E402

MODEL = settings.HF_MODEL


def _model_cached() -> bool:
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _model_cached(),
    reason=f"requer GPU CUDA e {MODEL} no cache local do Hugging Face (não baixa nada)",
)


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def fp32():
    from inference_service.backends.hf import HuggingFaceBackend

    backend = HuggingFaceBackend(quant="fp32", local_files_only=True)
    yield backend
    del backend
    _free()


def test_single_generation_reports_tokens_ttft_and_counters(fp32):
    fp32.reset_counters()
    (res,) = fp32.generate_batch([GenerationRequest("What is 2 + 2? Answer with just the number.", 8)])
    assert res.text.strip() != ""
    assert 0 < res.completion_tokens <= 8
    assert res.prompt_tokens > 5
    assert res.ttft_ms is not None and res.ttft_ms > 0
    assert res.finish_reason in {"stop", "length"}
    assert fp32.forward_pass_count == 1 and fp32.total_items_processed == 1
    assert fp32.tokens_generated_total == res.completion_tokens


def test_left_padded_batch_matches_single_greedy_generation(fp32):
    """Prova que o padding à esquerda + máscara não corrompe a geração:
    o mesmo prompt gera o mesmo texto sozinho e dentro de um lote com prompts
    de tamanhos bem diferentes."""
    prompts = [
        "Qual é a capital do Brasil?",
        "Explique em uma frase o que é batching dinâmico de requisições em servidores de inferência.",
        "Liste três cores primárias.",
    ]
    singles = [fp32.generate_batch([GenerationRequest(p, 12)])[0].text for p in prompts]
    batched = fp32.generate_batch([GenerationRequest(p, 12) for p in prompts])
    assert [r.text for r in batched] == singles


def test_each_sequence_respects_its_own_token_budget(fp32):
    short, long_ = fp32.generate_batch(
        [
            GenerationRequest("Conte de 1 até 50 separando por vírgula.", 3),
            GenerationRequest("Conte de 1 até 50 separando por vírgula.", 14),
        ]
    )
    assert short.completion_tokens == 3 and short.finish_reason == "length"
    assert 3 < long_.completion_tokens <= 14


def test_cancelled_sequence_stops_generating_but_batch_mates_finish(fp32):
    cancel = [threading.Event(), threading.Event()]
    cancel[0].set()  # cancelada antes do primeiro passo de decodificação
    a, b = fp32.generate_batch(
        [
            GenerationRequest("Conte de 1 até 50 separando por vírgula.", 24),
            GenerationRequest("Conte de 1 até 50 separando por vírgula.", 24),
        ],
        cancel,
    )
    assert a.cancelled is True and a.finish_reason == "cancelled"
    assert a.completion_tokens <= 1  # parou no primeiro passo
    assert b.cancelled is False and b.completion_tokens > 5


def test_quality_probe_returns_finite_perplexity_and_leaves_counters_alone(fp32):
    fp32.reset_counters()
    q = fp32.evaluate_quality(max_new_tokens=10)
    assert q["quant"] == "fp32" and q["perplexity_tokens"] > 20
    assert 1.0 < q["perplexity"] < 1e4
    assert len(q["greedy_outputs"]) == len(q["greedy_prompts"]) == 8
    assert fp32.forward_pass_count == 0 and fp32.total_items_processed == 0


def test_describe_reports_real_memory_and_precision(fp32):
    d = fp32.describe()
    assert d["quant"] == "fp32" and d["bits"] == 32 and d["device"].startswith("cuda")
    assert d["parameters"] > 400_000_000
    assert d["memory_footprint_bytes"] > 1_500_000_000  # ~1,9 GB em float32
    assert d["gpu_peak_allocated_bytes"] >= d["memory_footprint_bytes"]


def test_llm_app_batches_concurrent_http_requests_on_the_real_model(fp32):
    from fastapi.testclient import TestClient

    from inference_service.llm_app import create_app
    from inference_service.llm_batching import LLMBatcher

    fp32.reset_counters()
    batcher = LLMBatcher(fp32, max_batch_size=8, max_wait_ms=400)
    with TestClient(create_app(backend=fp32, batcher=batcher)) as client:
        def call(i):
            return client.post(
                "/generate", json={"prompt": f"Diga o número {i} por extenso.", "max_new_tokens": 8}
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(call, range(4)))
        health = client.get("/health").json()
    assert all(r.status_code == 200 and r.json()["completion_tokens"] > 0 for r in responses)
    assert all(r.json()["ttft_ms"] and r.json()["ttft_ms"] > 0 for r in responses)
    assert health["items_processed"] == 4
    assert health["batches_executed"] < 4  # batching real no modelo real


@pytest.mark.parametrize("mode", ["int8", "nf4"])
def test_bitsandbytes_modes_load_generate_and_shrink_memory(mode, fp32):
    pytest.importorskip("bitsandbytes")
    from inference_service.backends.hf import HuggingFaceBackend

    fp32_bytes = fp32.describe()["memory_footprint_bytes"]
    backend = HuggingFaceBackend(quant=mode, local_files_only=True)
    try:
        started = time.perf_counter()
        (res,) = backend.generate_batch([GenerationRequest("Qual é a capital do Brasil?", 10)])
        assert time.perf_counter() - started < 120
        assert res.text.strip() != "" and res.completion_tokens > 0
        d = backend.describe()
        assert d["quant"] == mode and d["bits"] == {"int8": 8, "nf4": 4}[mode]
        assert d["memory_footprint_bytes"] < 0.5 * fp32_bytes
    finally:
        del backend
        _free()
