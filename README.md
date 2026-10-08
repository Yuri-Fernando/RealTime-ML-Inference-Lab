# RealTime-ML-Inference-Lab

PyTorch + FastAPI/WebSocket (async) + admission control + batching dinamico + cancelamento real + cache + backpressure + telemetria GPU real + Prometheus/Grafana (validado rodando via Docker) + load test com concurrency sweep real (P50/P95/P99 por nível + ponto de saturação medido) + adapters HTTP reais para Triton/vLLM (sem backend real disponivel nesta sessao)

## Status

🟡 **MVP real e testado na parte de serving (M0-M3 da sessão inicial) + M2/M3/M6/M7 aprofundados nesta sessão (admission control explícito, cancelamento real, telemetria GPU real, observability validada rodando, capacity sweep medido); adapters Triton/vLLM reais porém sem backend real testado end-to-end (M4/M5 fora de escopo desta sessão).**

O que é **real** e verificado nesta sessão (2026-10-08) e na anterior (2026-10-07):
- GPU real disponível no ambiente de desenvolvimento: `torch.cuda.is_available() == True`, NVIDIA GeForce **RTX 3060 Ti** (compute capability 8.6). O serviço roda o modelo nessa GPU de verdade (`device: "cuda"` confirmado via `/health`), com fallback automático para CPU caso não haja GPU (`RTML_FORCE_CPU=1` força CPU para testes/comparação).
- O modelo servido (`inference_service/model.py`) é uma CNN pequena e real (PyTorch), com **pesos aleatórios não treinados** — o objetivo deste repo é a infraestrutura de serving (batching, cache, backpressure, admissão, cancelamento, métricas), não a acurácia do modelo.
- **Admission control explícito** (`inference_service/admission.py`): a decisão "capacity? -> queue/execute : reject/defer" (seção 11 do design doc) foi extraída do `DynamicBatcher` para um módulo próprio, testável isoladamente sem asyncio/modelo/fila (`tests/test_admission.py`). O `DynamicBatcher` agora delega a essa classe em vez de checar o limite inline.
- **Cancelamento real** (`inference_service/batching.py`, `DynamicBatcher.cancel()`): um request ainda na fila (não processado pelo worker) é marcado e **excluído do tensor do batch antes do forward pass** — não chega à GPU/CPU. Provado em `tests/test_cancellation.py` lendo `model_backend.forward_pass_count`/`total_items_processed` (ficam em 0 para o request cancelado). Exposto via `POST /cancel/{request_id}` (HTTP) e via desconexão real de WebSocket (`tests/test_app_integration.py::test_websocket_disconnect_cancels_inflight_request`, que fecha a conexão de verdade e confirma o contador `rtml_cancellations_total{reason="client_disconnect"}` incrementou).
- **Telemetria GPU real** (`inference_service/gpu_telemetry.py`): `torch.cuda.memory_allocated()`/`memory_reserved()` (sempre disponíveis com CUDA) e `torch.cuda.utilization()` (requer `nvidia-ml-py`, instalado nesta sessão — sem isso, `torch.cuda.utilization()` levanta `RuntimeError`, confirmado e corrigido instalando o pacote). Lido a cada scrape de `/metrics` (não é um sampler em background com atraso), nunca estimado — se a leitura falhar, a métrica simplesmente não é publicada naquele scrape em vez de publicar um valor inventado.
- Batching dinâmico, cache LRU, admissão e backpressure são comportamento real, coberto por **40 testes automatizados** (todos passando: os 23 originais + 7 de admissão + 6 de cancelamento + 4 de integração HTTP/WS novos) que rodam contra o event loop asyncio real e a GPU/CPU real — nada aqui é mock do comportamento testado.
- O load test (`loadtest/load_test.py`) gera tráfego HTTP concorrente real e mede latência real round-trip, agora com **modo de concurrency sweep** (`--sweep 1,2,4,8,16,32,64`) que roda o mesmo teste real em cada nível sequencialmente e escreve um relatório (`loadtest/reports/capacity-sweep-*.{json,md}`) com o ponto de saturação **medido** (não estimado por fórmula). Ver seção Resultados.
- **Prometheus + Grafana rodando de verdade via Docker Compose nesta sessão** — ver seção "Observability (validado)" abaixo para a evidência real capturada.

O que **não** foi testado de ponta a ponta nesta sessão:
- **Nenhum servidor NVIDIA Triton Inference Server ou vLLM real foi implantado.** Os adapters em `adapters/triton_vllm/` implementam os protocolos de wire reais e documentados (Triton KServe v2 HTTP, vLLM OpenAI-compatible HTTP) e são testados contra um **mock HTTP local in-process** (via `httpx.ASGITransport`), não contra um Triton/vLLM real. Isso é mock de uma dependência externa ausente, não do comportamento do cliente em si. Fora de escopo explícito desta sessão (M4/M5 do roadmap) — exigiria infraestrutura/modelo real que não foi provisionado aqui.
- Multi-GPU, NCCL, TensorRT, quantization, voz/WebRTC/SIP: fora de escopo desta sessão e da MVP em geral — ver `ROADMAP.md` V2/V3 e os NON-GOALS explícitos do master spec.

## Observability (validado)

Docker estava disponível neste ambiente (`docker --version` / `docker compose version` confirmados) mas o Docker Desktop não estava com o engine rodando — foi iniciado nesta sessão e o engine ficou pronto antes de subir a stack.

**Achado real de ambiente:** bind-mount direto do `observability/docker-compose.yml` (que vive dentro da pasta sincronizada pelo Google Drive Desktop, unidade `G:`) falhou — o Docker Desktop tentou montar o arquivo `prometheus.yml` e recebeu "not a directory" / tratou o arquivo como diretório, porque o filesystem virtual do Google Drive não se comporta como um filesystem local para bind mounts do Docker (mesma classe de problema documentada para `node_modules`/junctions em outros projetos do pool). Contorno aplicado: os arquivos de config (`docker-compose.yml`, `prometheus.yml`, provisioning do Grafana) foram copiados para uma pasta de staging local fora do Drive (`C:\tmp\rtml-observability`) e o `docker compose up -d` foi executado de lá, apontando para o `/metrics` real do serviço rodando no host via `host.docker.internal:8000`. Os arquivos-fonte no repo não mudaram — só o diretório de execução do compose precisa ficar fora do Drive.

Resultado real, verificado via API HTTP do Prometheus (evidência salva em `observability/evidence/`):
- `GET /api/v1/targets` → `health: "up"` para o job `realtime-ml-inference-lab`, scrape real de `http://host.docker.internal:8000/metrics` (`observability/evidence/prometheus_targets.json`).
- `GET /api/v1/query?query=rtml_gpu_available` → `1` (GPU real detectada pelo processo sendo raspado) (`observability/evidence/prometheus_query_gpu_available.json`).
- `GET /api/v1/query?query=rtml_requests_total` → contador real refletindo as requisições HTTP enviadas durante o teste (`observability/evidence/prometheus_query_requests_total.json`).
- `GET /api/v1/query?query=rtml_gpu_memory_allocated_bytes` → bytes reais alocados na GPU (`observability/evidence/prometheus_query_gpu_memory.json`).
- Grafana: datasource "Prometheus" provisionado automaticamente e apontando para `http://prometheus:9090` (`observability/evidence/grafana_datasource.json`), e o dashboard `RealTime-ML-Inference-Lab` (uid `rtml-inference-lab`) foi encontrado via `GET /api/search?type=dash-db` (`observability/evidence/grafana_dashboards.json`) — ou seja, o dashboard JSON provisionado carregou de verdade no Grafana rodando.
- Containers parados (`docker compose down`) ao final da verificação para não deixar serviços órfãos rodando.

Isso substitui o estado "não executado" da sessão anterior por "executado e verificado com evidência real" — mas é uma verificação pontual desta sessão, não uma garantia de que a stack continua montada (os containers foram removidos ao final, como é correto para não deixar processos de fundo).

## Descrição / Contexto

Gap identificado na auditoria de portfólio (seção "7) ML SYSTEMS / GPU / REAL-TIME INFERENCE"): nenhum repo do portfólio demonstrava infraestrutura de serving de ML de baixa latência — batching dinâmico, cache, backpressure, métricas de produção, client real para servidores de inferência especializados (Triton/vLLM). Este repo cobre esse gap com um MVP real e testável.

## Objetivo

Demonstrar, com código real e testado (não apenas documentação), as técnicas centrais de sistemas de inferência em tempo real:
- Serving assíncrono (FastAPI + WebSocket) sem bloquear o event loop por request individual.
- Admission control explícito, separado do batching (capacity? -> queue/execute : reject/defer).
- Batching dinâmico que agrupa requisições concorrentes em um único forward pass.
- Cancelamento real de requisições em andamento (sem consumir GPU, sem liberar recurso incorretamente).
- Cache de resultados determinístico.
- Backpressure real (rejeição controlada sob carga, não timeout silencioso).
- Telemetria de GPU real (VRAM, utilização) e observabilidade de produção (métricas Prometheus, dashboard Grafana) validada rodando via Docker.
- Capacity planning real via concurrency sweep medido (não estimado).
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
AdmissionController (inference_service/admission.py)
      │  capacity? -> admit : reject (decisão pura, testável isolada)
      ▼ admit
DynamicBatcher (inference_service/batching.py)
      │  asyncio.Queue + cancelamento real (request cancelado nunca stackeado no tensor)
      │  agrupa N requests em até max_wait_ms / max_batch_size
      ▼
ModelBackend (inference_service/model.py) — TinyCNN real em CUDA/CPU
      │
      ▼
Prometheus metrics (inference_service/metrics.py, + gpu_telemetry.py) ──► /metrics ──► Prometheus ──► Grafana

adapters/triton_vllm/ — clientes HTTP reais para Triton (KServe v2) e vLLM (OpenAI-compatible)
                         (sem backend real disponível nesta sessão — ver Status)
```

## Funcionamento

1. Uma requisição chega via `POST /infer` ou `WS /ws/infer` com um vetor de floats (`RTML_INPUT_DIM`, default 3×32×32 = 3072, formato CIFAR-like), recebendo um `request_id` (gerado ou fornecido pelo cliente).
2. O hash do input é verificado no cache (`inference_service/cache.py`, LRU real). Hit → retorna sem tocar o modelo.
3. Miss → `AdmissionController.decide(...)` (`inference_service/admission.py`) decide, com base no `in_flight` atual vs `RTML_MAX_QUEUE_DEPTH`, se a requisição é admitida. Rejeitada → HTTP 429 imediato (backpressure real, não timeout). Admitida → entra na fila do `DynamicBatcher`.
4. Um worker assíncrono único drena a fila: pega a primeira requisição disponível, continua absorvendo mais (até `RTML_MAX_BATCH_SIZE`) enquanto não passar `RTML_MAX_BATCH_WAIT_MS` desde a primeira. Isso é o batching dinâmico.
5. Antes do forward pass, qualquer requisição marcada como cancelada (via `POST /cancel/{request_id}` ou desconexão do WebSocket) é **filtrada do batch** — nunca é empilhada no tensor, nunca toca GPU/CPU.
6. O batch (já sem os cancelados) é empilhado em um tensor e passa por **um único** forward pass do modelo (GPU ou CPU). Cada requisição recebe sua fatia do resultado.
7. Resultado é salvo no cache e devolvido ao cliente. Métricas (latência, batch size, cache hit/miss, forward passes, profundidade de fila, cancelamentos, telemetria GPU) são atualizadas em `/metrics`.

## Admission Control

Extraído para `inference_service/admission.py` nesta sessão (antes vivia misturado dentro do `DynamicBatcher.submit()`). Segue o modelo da seção 11 do design doc:

```
request
  │
  ▼
capacity? (in_flight < max_queue_depth)
  ├─ yes → ADMIT  → entra na fila, segue para batching
  └─ no  → REJECT → HTTP 429 imediato
```

`AdmissionController.decide(current_in_flight)` é uma função pura sobre `(in_flight, max_queue_depth)` — sem fila, sem lock, sem modelo — testável em isolamento (`tests/test_admission.py`, 7 testes). O `DynamicBatcher` mantém o `in_flight` (incremento/decremento atômico sob lock) e apenas consulta a decisão.

## Cancelamento

Implementado em `inference_service/batching.py` (`DynamicBatcher.cancel(request_id)`), real e não cosmético:

- Cada requisição pendente é rastreada por `request_id` em `_pending_by_id`.
- `cancel(request_id)` marca a requisição como cancelada e resolve seu `Future` com `CancelledError` imediatamente — o chamador não espera a janela de batch terminar.
- Se o worker ainda não processou essa requisição, ela é **filtrada antes de montar o tensor** (`_process_batch`) — nunca chega ao `model_backend.infer_batch(...)`.
- Exposto por dois caminhos reais:
  - `POST /cancel/{request_id}`: cancelamento explícito via HTTP.
  - Desconexão de WebSocket: `WS /ws/infer` corre a inferência contra uma goroutine que escuta `websocket.receive()` esperando por `"websocket.disconnect"`; se o cliente desconectar enquanto a inferência está em andamento, `batcher.cancel(request_id, reason="client_disconnect")` é chamado antes que qualquer resultado seja produzido/consumido.
- Métrica real: `rtml_cancellations_total{reason="explicit"|"client_disconnect"}`.
- Prova real (não apenas lógica): `tests/test_cancellation.py` cancela uma requisição ainda na fila e confirma `model_backend.forward_pass_count == 0` e `total_items_processed == 0` — ou seja, a GPU/CPU nunca processou aquele item. `tests/test_app_integration.py::test_websocket_disconnect_cancels_inflight_request` fecha uma conexão WebSocket real enquanto uma requisição está em voo (janela de batch alargada para garantir a corrida) e confirma que o contador Prometheus de cancelamento por desconexão realmente incrementou.

## Telemetria de GPU

`inference_service/gpu_telemetry.py`, lida fresca a cada scrape de `/metrics` (não é um sampler de fundo com atraso):

- `torch.cuda.memory_allocated()` / `torch.cuda.memory_reserved()`: sempre disponíveis quando há CUDA, vêm direto do allocator da PyTorch.
- `torch.cuda.utilization()`: depende de `nvidia-ml-py` (pacote `pynvml`). **Achado real desta sessão**: sem esse pacote instalado, a chamada levanta `RuntimeError` ("nvidia-ml-py does not seem to be installed") — confirmado rodando no ambiente real antes de instalar o pacote. Corrigido instalando `nvidia-ml-py` (ver `requirements.txt`). Se a leitura falhar por qualquer motivo em outro ambiente, a métrica correspondente simplesmente não é publicada naquele scrape — nunca um valor inventado.
- Métricas expostas: `rtml_gpu_available`, `rtml_gpu_memory_allocated_bytes`, `rtml_gpu_memory_reserved_bytes`, `rtml_gpu_utilization_percent`.

## Modelagem

`TinyCNN` (`inference_service/model.py`): 2 blocos conv+relu+maxpool (3→16→32 canais) + linear final, entrada 3×32×32, saída softmax sobre `RTML_NUM_CLASSES` (default 10). Pesos inicializados com seed fixa (`RTML_MODEL_SEED`), **não treinados** — propositalmente, já que o foco é a infraestrutura de serving, não a tarefa de classificação em si.

## Experimental

- Ambiente de desenvolvimento: Windows, Python 3.10.8, PyTorch 2.13.0+cu126, GPU NVIDIA RTX 3060 Ti (confirmada via `torch.cuda.get_device_name(0)`), Docker 29.8.0 / Docker Compose v5.5.1 (engine iniciado manualmente nesta sessão).
- Suite de testes: **40 testes** (`pytest`), 100% passando — os 23 originais da sessão de criação + 7 novos de admission control (`tests/test_admission.py`) + 6 novos de cancelamento (`tests/test_cancellation.py`) + 4 novos de integração HTTP/WS (request_id, `/cancel`, telemetria GPU em `/metrics`, cancelamento real via desconexão de WebSocket).
- Load test real + concurrency sweep: ver seção Resultados.

## Tecnologias

PyTorch · FastAPI · Starlette/Uvicorn · WebSockets · Pydantic v2 · prometheus-client · nvidia-ml-py (telemetria GPU real) · httpx · pytest/pytest-asyncio · Docker Compose (Prometheus + Grafana — **rodado e validado nesta sessão**, ver "Observability (validado)") · Triton Inference Server (protocolo KServe v2, cliente real) · vLLM (API OpenAI-compatible, cliente real)

## Resultados

### Load test single-level (sessão anterior, mantido para referência)

`python loadtest/load_test.py --requests 2000 --concurrency 32` (pior caso: cada request com input aleatório diferente, sem reaproveitar cache):

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

### Concurrency sweep / capacity planning (nesta sessão)

`python loadtest/load_test.py --sweep 1,2,4,8,16,32,64 --requests-per-level 300` rodado contra a instância local real (GPU RTX 3060 Ti). Relatório completo em `loadtest/reports/capacity-sweep-20261008-141215.{json,md}`:

| Concorrência | Requests | Duração (s) | Throughput (req/s) | Erro | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 300 | 3.92 | 76.51 | 0.0% | 12.6 | 15.0 | 16.8 |
| 2 | 300 | 3.43 | 87.34 | 0.0% | 21.7 | 24.4 | 29.7 |
| 4 | 300 | 2.58 | 116.36 | 0.0% | 33.6 | 42.3 | 45.1 |
| 8 | 300 | 2.52 | **119.11 (pico)** | 0.0% | 64.4 | 89.5 | 159.3 |
| 16 | 300 | 5.61 | 53.45 | 0.0% | 205.7 | 783.7 | 1149.4 |
| 32 | 300 | 4.73 | 63.43 | 0.0% | 372.8 | 1223.1 | 1707.1 |
| 64 | 300 | 4.64 | 64.60 | 0.0% | 975.9 | 1425.3 | 1467.4 |

**Ponto de saturação medido (não estimado): concorrência = 8.** Entre concorrência 8 → 16, o throughput CAI de 119 para 53 req/s enquanto o P95 salta de 89ms para 784ms (quase 9x) — o joelho de fila clássico (queueing delay dominando a latência) observado diretamente neste hardware/modelo/configuração de batching, não inferido por fórmula. Zero erros em todos os níveis (o `RTML_MAX_QUEUE_DEPTH` default de 64 absorveu toda a carga testada sem rejeitar nada via 429 — a degradação é de latência de fila, não de admissão rejeitando). A regra de detecção de saturação usada pelo script (`_find_saturation_point` em `loadtest/load_test.py`) é: taxa de erro > 1% OU P95 mais que dobra vs. o nível anterior enquanto o throughput para de crescer (>15%).

## Roadmap

Ver `ROADMAP.md` para o detalhamento completo MVP / V2 / V3.

## Próximos Passos

1. Investigar a causa da degradação de latência entre concorrência 8 e 16 com profiling real (Nsight Systems / PyTorch Profiler) — hipótese provável é a mesma já documentada (overhead de cuDNN/sincronização CUDA por batch de tamanho variável), mas não medida diretamente ainda.
2. Obter acesso a um Triton Inference Server ou vLLM real (local ou cloud) para validar os adapters de ponta a ponta (M4/M5 do roadmap, não tentado nesta sessão por estar fora de escopo).
3. V2: TensorRT, quantization, CUDA Graphs (ver `ROADMAP.md`) — atacam diretamente o knee de latência medido acima.
4. Re-rodar a stack de observability com a stack do app rodando simultaneamente por mais tempo, para validar os painéis de GPU/cancelamento do dashboard sob tráfego sustentado (nesta sessão a verificação foi pontual, containers removidos ao final).

## Como rodar

```bash
pip install -r requirements.txt

# Rodar o serviço
uvicorn inference_service.app:app --host 127.0.0.1 --port 8000

# Rodar a suite de testes (40 testes)
pytest -q

# Rodar um load test de nível único
python loadtest/load_test.py --requests 2000 --concurrency 32

# Rodar um concurrency sweep (capacity planning real) — escreve em loadtest/reports/
python loadtest/load_test.py --sweep 1,2,4,8,16,32,64 --requests-per-level 300

# Cancelar uma requisição em andamento (explícito)
curl -X POST http://127.0.0.1:8000/cancel/<request_id>

# Observability (requer Docker Desktop com o engine rodando).
# Nota: se o repo estiver numa pasta sincronizada por Google Drive/OneDrive, o
# bind-mount do docker-compose pode falhar (ver seção "Observability (validado)").
# Copie observability/ para uma pasta local fora do Drive antes de subir:
cd observability && docker compose up -d
# Prometheus: http://localhost:9090  |  Grafana: http://localhost:3000 (admin/admin)
```

## Licença

MIT — ver `LICENSE`.
