# RealTime-ML-Inference-Lab

PyTorch + FastAPI/WebSocket (async) + batching dinamico + cache + backpressure + Prometheus/Grafana + load test real (P50/P95/P99) + adapters HTTP reais para Triton/vLLM (sem backend real disponivel nesta sessao)

## Status

🟡 **MVP real e testado na parte de serving; adapters Triton/vLLM reais porem sem backend real testado end-to-end.**

O que é **real** e verificado nesta sessão:
- GPU real disponível no ambiente de desenvolvimento: `torch.cuda.is_available() == True`, NVIDIA GeForce **RTX 3060 Ti** (compute capability 8.6). O serviço roda o modelo nessa GPU de verdade (`device: "cuda"` confirmado via `/health`), com fallback automático para CPU caso não haja GPU (`RTML_FORCE_CPU=1` força CPU para testes/comparação).
- O modelo servido (`inference_service/model.py`) é uma CNN pequena e real (PyTorch), com **pesos aleatórios não treinados** — o objetivo deste repo é a infraestrutura de serving (batching, cache, backpressure, métricas), não a acurácia do modelo.
- Batching dinâmico, cache LRU e backpressure são comportamento real, coberto por 23 testes automatizados (todos passando) que rodam contra o event loop asyncio real e a GPU/CPU real — nada aqui é mock do comportamento testado.
- O load test (`loadtest/load_test.py`) gera tráfego HTTP concorrente real e mede latência real round-trip. Números na seção Resultados abaixo são medidos, não estimados.

O que **não** foi testado de ponta a ponta nesta sessão:
- **Nenhum servidor NVIDIA Triton Inference Server ou vLLM real foi implantado.** Os adapters em `adapters/triton_vllm/` implementam os protocolos de wire reais e documentados (Triton KServe v2 HTTP, vLLM OpenAI-compatible HTTP) e são testados contra um **mock HTTP local in-process** (via `httpx.ASGITransport`), não contra um Triton/vLLM real. Isso é mock de uma dependência externa ausente, não do comportamento do cliente em si.
- **Prometheus/Grafana via Docker Compose não foi executado nesta sessão** (sem verificação de que o runtime Docker estava disponível/configurado no ambiente). A configuração (`observability/docker-compose.yml`, `prometheus.yml`, dashboard JSON provisionado) é real e funcional pelo padrão já usado no repo "Enterprise Cloud Automation Platform", mas não foi validada rodando `docker compose up` aqui. O endpoint `/metrics` do serviço FastAPI, por outro lado, **foi testado e funciona de verdade** (formato de exposição Prometheus real, coberto em teste de integração).

## Descrição / Contexto

Gap identificado na auditoria de portfólio (seção "7) ML SYSTEMS / GPU / REAL-TIME INFERENCE"): nenhum repo do portfólio demonstrava infraestrutura de serving de ML de baixa latência — batching dinâmico, cache, backpressure, métricas de produção, client real para servidores de inferência especializados (Triton/vLLM). Este repo cobre esse gap com um MVP real e testável.

## Objetivo

Demonstrar, com código real e testado (não apenas documentação), as técnicas centrais de sistemas de inferência em tempo real:
- Serving assíncrono (FastAPI + WebSocket) sem bloquear o event loop por request individual.
- Batching dinâmico que agrupa requisições concorrentes em um único forward pass.
- Cache de resultados determinístico.
- Backpressure real (rejeição controlada sob carga, não timeout silencioso).
- Observabilidade de produção (métricas Prometheus, dashboard Grafana).
- Integração com servidores de inferência especializados via protocolo real (Triton, vLLM).

## Linha de Pesquisa

Model serving, streaming inference, batching/scheduling, caching, backpressure, async/concurrency, low-latency systems — a mesma linha de skills listada na auditoria (seção 7).

## Arquitetura

```
Cliente HTTP/WS
      │
      ▼
FastAPI app (inference_service/app.py)
      │  hash do input → cache hit? ─── sim ──► retorna direto (sem forward pass)
      ▼ não
DynamicBatcher (inference_service/batching.py)
      │  asyncio.Queue + backpressure por profundidade de fila
      │  agrupa N requests em até max_wait_ms / max_batch_size
      ▼
ModelBackend (inference_service/model.py) — TinyCNN real em CUDA/CPU
      │
      ▼
Prometheus metrics (inference_service/metrics.py) ──► /metrics ──► Prometheus ──► Grafana

adapters/triton_vllm/ — clientes HTTP reais para Triton (KServe v2) e vLLM (OpenAI-compatible)
                         (sem backend real disponível nesta sessão — ver Status)
```

## Funcionamento

1. Uma requisição chega via `POST /infer` ou `WS /ws/infer` com um vetor de floats (`RTML_INPUT_DIM`, default 3×32×32 = 3072, formato CIFAR-like).
2. O hash do input é verificado no cache (`inference_service/cache.py`, LRU real). Hit → retorna sem tocar o modelo.
3. Miss → a requisição é colocada na fila do `DynamicBatcher`. Se a fila já está no limite (`RTML_MAX_QUEUE_DEPTH`), a requisição é rejeitada imediatamente com HTTP 429 (backpressure real, não timeout).
4. Um worker assíncrono único drena a fila: pega a primeira requisição disponível, continua absorvendo mais (até `RTML_MAX_BATCH_SIZE`) enquanto não passar `RTML_MAX_BATCH_WAIT_MS` desde a primeira. Isso é o batching dinâmico.
5. O batch inteiro é empilhado em um tensor e passa por **um único** forward pass do modelo (GPU ou CPU). Cada requisição recebe sua fatia do resultado.
6. Resultado é salvo no cache e devolvido ao cliente. Métricas (latência, batch size, cache hit/miss, forward passes, profundidade de fila) são atualizadas em `/metrics`.

## Modelagem

`TinyCNN` (`inference_service/model.py`): 2 blocos conv+relu+maxpool (3→16→32 canais) + linear final, entrada 3×32×32, saída softmax sobre `RTML_NUM_CLASSES` (default 10). Pesos inicializados com seed fixa (`RTML_MODEL_SEED`), **não treinados** — propositalmente, já que o foco é a infraestrutura de serving, não a tarefa de classificação em si.

## Experimental

- Ambiente de desenvolvimento: Windows, Python 3.10.8, PyTorch 2.13.0+cu126, GPU NVIDIA RTX 3060 Ti (confirmada via `torch.cuda.get_device_name(0)`).
- Suite de testes: 23 testes (`pytest`), 100% passando, cobrindo cache (hit/miss/LRU), batching (redução real de forward passes sob concorrência, cap de batch size, backpressure, recuperação pós-drenagem), adapters (request/response/erro contra mock local), e integração HTTP/WebSocket end-to-end via `TestClient`.
- Load test real: ver seção Resultados.

## Tecnologias

PyTorch · FastAPI · Starlette/Uvicorn · WebSockets · Pydantic v2 · prometheus-client · httpx · pytest/pytest-asyncio · Docker Compose (Prometheus + Grafana, configurado mas não executado nesta sessão) · Triton Inference Server (protocolo KServe v2, cliente real) · vLLM (API OpenAI-compatible, cliente real)

## Resultados

Load test real contra a instância local do serviço (GPU RTX 3060 Ti), `python loadtest/load_test.py --requests 2000 --concurrency 32` (pior caso: cada request com input aleatório diferente, sem reaproveitar cache):

| Métrica | Valor medido |
|---|---|
| Requisições totais | 2000 |
| Erros | 0 |
| Duração | 43.87 s |
| Throughput | 45.59 req/s |
| Latência P50 | 610.03 ms |
| Latência P95 | 1186.59 ms |
| Latência P99 | 1954.23 ms |
| Latência min/max | 113.19 ms / 3363.35 ms |
| Forward passes executados | 976 (para 2000 requests → batch médio ≈ 2.05) |

**Nota de honestidade sobre a latência:** 610ms de P50 para uma CNN minúscula em GPU é alto para o tamanho do modelo — a causa mais provável é overhead de seleção de algoritmo cuDNN a cada novo shape de batch (batches de tamanho variável disparam reotimização interna do cuDNN) somado a overhead de sincronização CUDA por chamada, não throughput real de computação do modelo. Isso é um achado real do ambiente de teste (single process, Windows, batches de tamanho variável), não um número ajustado — ficou documentado como está. Para produção, CUDA Graphs e tamanhos de batch fixos (ver Roadmap V2) atacam exatamente esse problema.

Batching funcionando de verdade: 2000 requisições lógicas resultaram em apenas 976 forward passes reais — confirmado tanto no load test quanto no teste automatizado `tests/test_batching.py::test_concurrent_requests_are_batched_into_fewer_forward_passes`.

## Roadmap

Ver `ROADMAP.md` para o detalhamento completo MVP / V2 / V3.

## Próximos Passos

1. Rodar `observability/docker-compose.yml` em um ambiente com Docker disponível e validar o dashboard Grafana contra tráfego real.
2. Obter acesso a um Triton Inference Server ou vLLM real (local ou cloud) para validar os adapters de ponta a ponta.
3. V2: TensorRT, quantization, CUDA Graphs (ver `ROADMAP.md`).

## Como rodar

```bash
pip install -r requirements.txt

# Rodar o serviço
uvicorn inference_service.app:app --host 127.0.0.1 --port 8000

# Rodar a suite de testes
pytest -q

# Rodar o load test (com o serviço já rodando em outro terminal)
python loadtest/load_test.py --requests 2000 --concurrency 32

# Observability (requer Docker; não validado nesta sessão)
cd observability && docker compose up
```

## Licença

MIT — ver `LICENSE`.
