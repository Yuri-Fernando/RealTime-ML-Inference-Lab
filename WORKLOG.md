# Worklog — Sessão de criação do RealTime-ML-Inference-Lab

**Data:** 2026-10-07

## Contexto

Repo novo recomendado pela auditoria de portfólio (seções 7 e 16) para cobrir o gap de
infraestrutura de inferência de ML em tempo real/GPU. Escopo desta sessão: MVP apenas
(V2/V3 documentados em `ROADMAP.md`, não implementados).

## O que foi feito, em ordem

1. Criada a pasta do repo e `git init` (sem worktree isolada, por instrução — worktrees
   travam neste ambiente por sincronização do Google Drive).
2. Verificação de ambiente antes de qualquer suposição:
   - `torch.cuda.is_available()` → **True**. GPU real: NVIDIA GeForce RTX 3060 Ti
     (compute capability 8.6). Matmul real testado na GPU com sucesso.
   - Isso contradiz a hipótese de partida da tarefa ("provavelmente sem GPU") — o
     ambiente real tem GPU disponível, então o backend de modelo roda em CUDA de
     verdade, não apenas CPU. Documentado honestamente no README.
   - Pacotes já instalados no Python 3.10 canônico: torch 2.13.0+cu126, fastapi
     0.136.3, uvicorn, prometheus_client, pytest 9.0.1, httpx 0.28.1, websockets,
     pydantic 2.12.5, starlette 0.49.3, pytest-asyncio 1.3.0. `respx`/`pytest-httpx`
     não instalados → optei por `httpx.ASGITransport` (built-in) para mockar
     servidores Triton/vLLM locais nos testes, sem dependências extras.
3. Implementado `inference_service/`: config, model (TinyCNN real, pesos não
   treinados, device CUDA/CPU automático), cache (LRU real por hash), batching
   (DynamicBatcher real com fila asyncio + backpressure), metrics (Prometheus real),
   app (FastAPI + WebSocket).
4. Implementado `adapters/triton_vllm/`: clientes HTTP reais para os protocolos
   documentados do Triton (KServe v2) e vLLM (OpenAI-compatible), sem depender de
   SDKs específicos (`tritonclient` etc. não estavam instalados).
5. Primeira rodada de testes: 7 falhas em `tests/test_app_integration.py` por
   `asyncio.Queue`/`asyncio.Lock` bindados ao event loop errado (singletons de
   módulo reutilizados entre sessões de `TestClient`, cada uma com seu próprio
   loop). Corrigido tornando a criação da queue/lock lazy (dentro de `start()` /
   `_ensure_initialized()`), e resetando-os em `stop()`.
6. Segunda rodada: 3 falhas por cache compartilhado entre testes (um teste
   "contaminava" o cache de outro ao usar o mesmo input). Corrigido com fixture
   `autouse` que limpa `result_cache` e reseta contadores do modelo antes/depois
   de cada teste.
7. Terceira rodada: **23 testes passando.**
8. Rodado o serviço de verdade (`uvicorn`) e o load test real
   (`loadtest/load_test.py --requests 2000 --concurrency 32`) contra ele.
   Resultado medido: P50 610ms / P95 1187ms / P99 1954ms, throughput 45.6 req/s,
   0 erros, 976 forward passes para 2000 requisições (batching real reduzindo
   quase à metade). Latência mais alta do que o esperado para uma CNN minúscula em
   GPU — hipótese documentada no README (overhead de seleção de algoritmo cuDNN por
   batch de tamanho variável), sem forjar um número "melhor".
9. Observabilidade: `docker-compose.yml` (Prometheus + Grafana) e dashboard JSON
   provisionado, seguindo o padrão do repo "Enterprise Cloud Automation Platform".
   **Não executado nesta sessão** (sem verificação de runtime Docker disponível) —
   documentado como tal.
10. README, CHANGELOG (0.1.0), ROADMAP (MVP/V2/V3), LICENSE (MIT), `.gitignore`,
    `requirements.txt`.
11. Commits incrementais em português ao final de cada módulo concluído.

## Decisões técnicas registradas

- Modelo: CNN pequena com pesos aleatórios (não treinada) — o repo demonstra
  infraestrutura de serving, não uma tarefa de ML específica. Documentado
  explicitamente para não implicar acurácia real.
- Mock de Triton/vLLM: mock de uma dependência EXTERNA ausente (não existe
  servidor real disponível), não do comportamento do cliente sob teste — distinção
  mantida consistente com a régua de honestidade do restante do portfólio.
- Cache key: hash SHA-256 do vetor de input arredondado (6 casas decimais) para
  tolerar ruído de ponto flutuante entre serializações JSON idênticas.

---

# Worklog — Sessão de aprofundamento (auditoria de maturidade)

**Data:** 2026-10-08

## Contexto

Especificação mais madura fornecida pelo usuário em dois arquivos fora do repo
(`03_RealTime-ML-Inference-Lab.md`, `RealTime_ML_Inference_Lab_Master_Spec.txt`),
refinando o escopo original. Tarefa: aprofundar M2/M3/M6/M7 do roadmap (admission
control explícito, cancelamento real, telemetria GPU real, observability
executada de verdade, load test com concurrency sweep) sem tentar M4/M5
(vLLM/Triton reais) nem V2/V3 (fora de escopo explícito).

Working tree confirmado limpo no início (2 commits já existiam da sessão anterior).

## O que foi feito, em ordem

1. Lidos README/CHANGELOG/ROADMAP/WORKLOG existentes + as duas specs novas na
   íntegra (seções "01 — PROJECT IDENTITY AND BOUNDARIES" e "SUCCESS DEFINITION"/
   "NON-GOALS FOR MVP" do master spec, conforme instruído, mais o resto como
   referência).
2. **Admission control** (`inference_service/admission.py`): extraída a decisão
   "capacity? -> admit : reject" do `DynamicBatcher.submit()` para
   `AdmissionController.decide(in_flight)`, função pura sem asyncio/fila/modelo.
   `DynamicBatcher` passou a delegar a ela; `max_queue_depth` virou uma property
   que repassa para o controller (mantendo compatibilidade com testes/código que
   já faziam `batcher.max_queue_depth = N`).
3. **Cancelamento real** (`inference_service/batching.py`): `PendingRequest`
   ganhou `request_id` e flag `cancelled`; `DynamicBatcher.cancel(request_id)`
   marca a flag e resolve o Future com `CancelledError` imediatamente;
   `_process_batch` filtra itens cancelados ANTES de montar o tensor — nunca
   chegam ao `model_backend.infer_batch(...)`. Nova exceção de domínio
   `RequestCancelledError` distingue "este request foi cancelado por nós" de
   uma cancelação externa genuína da task (para não engolir cancelamentos
   legítimos do asyncio). Endpoint `POST /cancel/{request_id}` adicionado;
   `WS /ws/infer` reescrito para correr a inferência contra um watcher de
   desconexão real (`websocket.receive()` em loop até `"websocket.disconnect"`),
   cancelando o request em voo se o cliente desconectar antes da resposta.
4. **Telemetria GPU** (`inference_service/gpu_telemetry.py`): testado
   `torch.cuda.memory_allocated()`/`memory_reserved()` (funcionam direto) e
   `torch.cuda.utilization()` — **falhou na primeira tentativa** com
   `RuntimeError: nvidia-ml-py does not seem to be installed`, confirmado antes
   de qualquer correção. Instalado `nvidia-ml-py` via pip no Python canônico;
   reconfirmado que `torch.cuda.utilization()` passou a retornar um valor real
   (7%, depois 25% sob carga simulada). Métricas novas no `/metrics`:
   `rtml_gpu_available`, `rtml_gpu_memory_allocated_bytes`,
   `rtml_gpu_memory_reserved_bytes`, `rtml_gpu_utilization_percent`,
   `rtml_cancellations_total{reason}`. Lidas a cada scrape, nunca estimadas.
5. Testes novos: `tests/test_admission.py` (7), `tests/test_cancellation.py` (6),
   mais 4 testes de integração em `tests/test_app_integration.py` (request_id,
   `/cancel` desconhecido, telemetria no `/metrics`, cancelamento real via
   desconexão de WebSocket fechando a conexão de verdade e lendo o contador
   Prometheus depois). Suite completa: 23 → 40 testes, todos passando
   (confirmado rodando `pytest -q` mais de uma vez durante a sessão).
6. **Observability — tentativa real de rodar via Docker**:
   - `docker --version` / `docker compose version` OK, mas `docker info` falhou
     (engine do Docker Desktop não estava rodando). Iniciado
     `Docker Desktop.exe` manualmente; engine ficou pronto em ~5s.
   - Serviço real iniciado via `uvicorn` (bind `0.0.0.0:8000`, confirmado
     `device: "cuda"` em `/health`), tráfego real gerado via PowerShell
     (`Invoke-RestMethod`) para popular métricas antes do scrape.
   - `docker compose up -d` direto na pasta do repo (dentro do Google Drive
     Desktop, unidade `G:`) **falhou**: "mount ... not a directory" ao tentar
     bind-mountar `prometheus.yml` — o filesystem virtual do Google Drive não
     se comporta como filesystem local para bind mounts (mesma classe de
     problema documentada em `rules/node-env.md` para `node_modules`).
   - Contornado copiando os arquivos de config (`docker-compose.yml`,
     `prometheus/`, `grafana/`) para `C:\tmp\rtml-observability` (fora do
     Drive) e rodando `docker compose up -d` de lá, apontando para
     `host.docker.internal:8000`.
   - Confirmado via API HTTP real do Prometheus: `GET /api/v1/targets` →
     `health: "up"`; `GET /api/v1/query?query=rtml_gpu_available` → `1`;
     `rtml_requests_total` e `rtml_gpu_memory_allocated_bytes` com valores
     reais. Grafana: datasource "Prometheus" provisionado
     (`http://prometheus:9090`) confirmado via `GET /api/datasources`;
     dashboard `rtml-inference-lab` confirmado carregado via
     `GET /api/search?type=dash-db`. Evidência JSON salva em
     `observability/evidence/` (4 arquivos).
   - Dois painéis novos adicionados ao dashboard JSON (memória de GPU,
     utilização de GPU + cancelamentos/s) — não re-verificados visualmente
     (Grafana já estava parado quando os painéis foram adicionados), mas a
     sintaxe das queries PromQL segue o mesmo padrão dos painéis já validados.
   - `docker compose down` ao final para não deixar containers órfãos.
7. **Load test — concurrency sweep real**: `loadtest/load_test.py` estendido
   com `--sweep` (níveis separados por vírgula) e `--requests-per-level`,
   rodando o mesmo teste real sequencialmente em cada nível contra a mesma
   instância do serviço, com pausa de 0.5s entre níveis. Detecção de saturação
   (`_find_saturation_point`): erro > 1% OU P95 mais que dobra vs. nível
   anterior enquanto throughput cresce menos de 15%. Rodado de verdade:
   `--sweep 1,2,4,8,16,32,64 --requests-per-level 300` contra a GPU real.
   Resultado medido: throughput sobe de 76→119 req/s entre concorrência 1→8,
   depois CAI para 53 req/s em concorrência 16 enquanto P95 salta de 89ms para
   784ms — saturação real identificada em concorrência 8. Relatório JSON+MD
   escrito em `loadtest/reports/capacity-sweep-20261008-141215.*`.
8. README/CHANGELOG/ROADMAP atualizados com os achados reais (nova seção
   "Observability (validado)" no README com a evidência e o contorno do
   bind-mount; CHANGELOG 0.2.0; ROADMAP com M2/M3/M6/M7 marcados avançados e
   M4/M5 explicitamente não tentados).

## Decisões técnicas registradas (sessão 2026-10-08)

- `AdmissionController` não possui estado próprio sobre `in_flight` — o
  `DynamicBatcher` mantém o contador sob lock e apenas consulta a decisão,
  porque o incremento precisa ser atômico com a decisão (mesma seção crítica).
  Separar a regra da contagem foi a forma de isolar a regra sem quebrar essa
  atomicidade.
- Cancelamento distingue "cancelamos este request de propósito" (gera
  `RequestCancelledError`, um erro de domínio) de uma cancelação assíncrona
  genuína da task por outro motivo (deixa `CancelledError` normal propagar) —
  checando a flag `request.cancelled` antes de converter a exceção.
- O WebSocket não tenta suportar múltiplas requisições concorrentes na mesma
  conexão (fora de escopo) — o protocolo continua um request por vez; a
  corrida implementada é apenas contra desconexão, não contra uma segunda
  mensagem de input chegando enquanto a primeira está em voo.
- Observability: a correção de bind-mount via staging local é um padrão já
  documentado no control plane do usuário (`rules/node-env.md`) para outro
  problema (npm install) — mesma causa raiz (filesystem virtual do Google
  Drive Desktop não suporta certas operações nativas do Windows/Docker),
  aplicada aqui para bind mounts do Docker.

---

# Sessão 3 — 2026-10-21 — V0.3: geração de texto e quantização

## Objetivo
Backends plugáveis (TinyCNN / HuggingFace / vLLM), serviço de geração com o mesmo
batching/admissão/cancelamento, quantização real (FP32/FP16/BF16/INT8/NF4) medida com o
mesmo load test, painel Grafana, e uma tentativa limitada de vLLM real.

## Passo a passo
1. Leitura do repositório (`model.py`, `batching.py`, `app.py`, `metrics.py`, `loadtest/`, adapters).
   Decisão: **não** mexer no caminho do TinyCNN; criar `llm_app.py`/`llm_batching.py` em paralelo,
   reaproveitando `AdmissionController` e as exceções do batcher histórico.
2. Sondagem do ambiente antes de projetar: Qwen2.5-0.5B no cache `Q:\Hugginface`; fp32/fp16/bf16/int8/nf4
   carregam e geram na RTX 3060 Ti (pesos medidos: 1885/942/942/601/430 MiB). `docker run --gpus all` ok.
3. Início do `docker pull vllm/vllm-openai` em segundo plano (limite de 15 min).
4. Implementação: backends, `LLMBatcher`, `llm_app`, métricas, extensão do `load_test.py`.
5. Testes. 27 verdes no primeiro corte (factory + batcher). O primeiro teste com o modelo real achou um
   **bug meu**: `finish_reason="stop"` onde devia ser `length` — tokens de padding após o orçamento da
   sequência eram lidos como EOS. Corrigido (só os primeiros `budget` tokens pertencem à sequência).
   Um teste meu esperava `cuda:0` e o dispositivo é `cuda`: corrigido o teste, não o código.
6. **vLLM**: o pull completou 8 de 40 camadas em 15 min (`exit=124`) → BLOCKED_BY_CONFIGURATION. Não repeti.
7. Benchmark, rodada 1: mostrou baseline de GPU ocioso de 94% e 6–7 GB em uso por **outros processos** e
   um P95 de 343 s no FP16 → **contaminada**. Preservada em `benchmarks/results/contaminated-20261021-150837/`.
   O processo do benchmark morreu depois no INT8 (`ReadError` no `/quality`, servidor sem traceback). Reproduzi
   o INT8 em thread principal, em thread secundária e como servidor manual: tudo passou. Causa não identificada.
8. Runner endurecido: espera a GPU ficar ociosa, flags de contaminação, tentativas, `--resume`,
   `PYTHONFAULTHANDLER`. Rodada 2 (limpa): 5 modos concluídos, todos marcados "medição limpa".
9. Suíte completa na cópia de staging em `D:` (o import de torch a partir do Drive é lento):
   `94 passed in 164.54s`, 0 pulados.

## Decisões
- Geração roda em `asyncio.to_thread`: o event loop continua atendendo admissão e cancelamento.
- Cancelamento durante a geração = `threading.Event` por sequência lido por um `StoppingCriteria`
  por sequência a cada passo; a sequência para, o lote termina quando todas pararam.
- vLLM = `supports_batching=False`: o servidor faz o batching contínuo; o cliente só aplica admissão.
- Sem cache de resultados no caminho LLM (prompts repetidos inflariam o benchmark).
- Testes e benchmark rodam numa cópia em `D:\Claude Temp\stagingealtime-v03epo`
  (mesma causa raiz da lição de bind-mount do Drive: filesystem virtual lento/incompatível).

## Pendências
- vLLM real (baixar a imagem com tempo/rede suficientes e rodar o mesmo sweep).
- Repetir o benchmark algumas vezes para estimar a variância de FP32/FP16/BF16.
- TTFT por streaming.
