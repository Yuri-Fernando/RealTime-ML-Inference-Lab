# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versionamento [SemVer](https://semver.org/lang/pt-BR/).

## [0.3.0] - 2026-10-21

Geração de texto com backends plugáveis e quantização real medida. O caminho
histórico (TinyCNN, `inference_service/app.py`) não foi alterado; os 40 testes
da v0.2.0 continuam passando sem modificação.

### Adicionado
- `inference_service/backends/`: protocolo `InferenceBackend`/`TextBackend`,
  `HuggingFaceBackend` (Qwen2.5-0.5B-Instruct em fp32/fp16/bf16/int8/nf4 via
  bitsandbytes; lote com padding à esquerda, orçamento de tokens e cancelamento
  por sequência, TTFT, sonda de qualidade), `VLLMBackend` (cliente do servidor
  vLLM, reaproveita `adapters/triton_vllm/vllm_client.py`) e `create_backend()`
  por `RTML_BACKEND` (`tinycnn` | `hf` | `vllm`). `TinyCNNBackend` é o
  `ModelBackend` histórico com outro nome (alias), sem fork.
- `inference_service/llm_batching.py` (`LLMBatcher`) e `llm_app.py`
  (`POST /generate`, `/cancel/{id}`, `/quality`, `/health`, `/metrics`):
  admissão (429), cancelamento real na fila, durante a geração (evento por
  sequência avaliado a cada passo) e na desconexão do cliente (499), timeout
  que também aborta a geração, `X-Request-ID` para cancelar por id.
- `inference_service/llm_metrics.py`: `rtml_llm_*` (latência, TTFT, tokens,
  tamanho/duração do lote, tokens/s, memória do modelo, pico de VRAM alocada,
  informação do backend).
- `loadtest/load_test.py --scenario llm [--max-new-tokens N]`: mede tokens/s e
  TTFT no mesmo concurrency sweep (relatório JSON/MD com colunas extras).
  O cenário padrão (`cnn`) é idêntico ao anterior.
- `benchmarks/run_quant_benchmark.py`: sobe o serviço real por modo, espera a
  GPU compartilhada ficar ociosa, roda o sweep, amostra `nvidia-smi`, coleta
  perplexidade e saídas greedy e gera relatório + gráficos; `--resume`,
  `--attempts`, sinalização de medição contaminada. Resultados:
  `benchmarks/results/quant-20261021-193218.{json,md}` e PNGs.
- Observabilidade: job de scrape `realtime-ml-inference-lab-llm` (porta 8100) e
  `observability/grafana/dashboards/llm-dashboard.json`.
- `docker-compose.vllm.yml` e `scripts/run_vllm_wsl.sh` (vLLM via Docker com GPU).
- 54 testes novos (94 no total): `test_backends_factory`, `test_llm_batching`
  (13), `test_llm_app`, `test_vllm_backend`, `test_hf_backend` (modelo real).

### Corrigido
- `HuggingFaceBackend`: tokens de padding gerados depois do orçamento de uma
  sequência eram lidos como EOS genuíno, então `finish_reason` saía `stop` em vez
  de `length` (pego pelo teste de orçamento por sequência, antes de qualquer
  benchmark).

### Notas de honestidade
- **vLLM real: BLOCKED_BY_CONFIGURATION.** `docker run --gpus all` funciona, mas o
  `docker pull vllm/vllm-openai:latest` concluiu 8 de 40 camadas no limite de 15
  min; a imagem não ficou disponível. Nenhum benchmark de vLLM real existe.
- A primeira rodada completa do benchmark foi **contaminada** por outros
  processos na GPU compartilhada e descartada como comparação (preservada em
  `benchmarks/results/contaminated-20261021-150837/`, com `NOTA.md`).
- Uma rodada intermediária morreu no `/quality` do INT8 (conexão encerrada pelo
  servidor, sem traceback); não reproduziu isoladamente (modelo INT8 em thread
  principal, em thread secundária e servidor manual funcionaram) e a rodada
  seguinte passou sem alterações no backend. Causa não identificada — provável
  interferência de outros processos na GPU; o runner agora registra tentativas e
  a cauda do log do servidor.
- Comparar FP32 × FP16 × BF16 por latência/vazão **não é sustentado** por esta
  medição (uma rodada, 24 requisições por nível, formação de lote variável).

## [0.2.0] - 2026-10-08

Auditoria/aprofundamento da MVP original contra a spec mais madura fornecida
(`03_RealTime-ML-Inference-Lab.md` + `RealTime_ML_Inference_Lab_Master_Spec.txt`).
Foco em M2/M3/M6/M7 do roadmap, com profundidade real em vez de espalhar fino.

### Adicionado
- `inference_service/admission.py`: `AdmissionController`, decisão de admissão
  ("capacity? -> queue/execute : reject/defer", seção 11 do design doc) extraída
  do `DynamicBatcher` para um módulo próprio, puro e testável sem asyncio/fila/modelo.
- Cancelamento real de requisições (`DynamicBatcher.cancel()`, `RequestCancelledError`):
  uma requisição cancelada enquanto ainda está na fila é filtrada do batch antes do
  forward pass — nunca toca GPU/CPU. Exposto via `POST /cancel/{request_id}` (HTTP) e
  via desconexão real de WebSocket em `WS /ws/infer` (corrida real entre a inferência
  em andamento e um watcher de desconexão, não um timeout arbitrário).
- `inference_service/gpu_telemetry.py`: telemetria real de GPU (`torch.cuda.memory_allocated`,
  `memory_reserved`, `utilization()` via `nvidia-ml-py`), lida a cada scrape de `/metrics`
  (não um sampler de fundo com atraso); métricas Prometheus novas: `rtml_gpu_available`,
  `rtml_gpu_memory_allocated_bytes`, `rtml_gpu_memory_reserved_bytes`,
  `rtml_gpu_utilization_percent`, `rtml_cancellations_total{reason}`.
- `loadtest/load_test.py --sweep <levels> --requests-per-level <n>`: concurrency sweep real
  (mesmo teste de carga rodado sequencialmente em cada nível contra a mesma instância),
  com detecção real de ponto de saturação (erro > 1% OU P95 mais que dobra enquanto
  throughput para de crescer) e relatório machine-generated em `loadtest/reports/*.{json,md}`.
- `tests/test_admission.py` (7 testes), `tests/test_cancellation.py` (6 testes), e 4 novos
  testes de integração em `tests/test_app_integration.py` (request_id na resposta, endpoint
  `/cancel`, telemetria GPU exposta em `/metrics`, cancelamento real via desconexão de
  WebSocket) — suite total: 23 → **40 testes**, 100% passando.
- Dois painéis novos no dashboard Grafana (`observability/grafana/dashboards/inference-dashboard.json`):
  memória de GPU (allocated/reserved) e utilização de GPU + cancelamentos/s.
- `requirements.txt`: `nvidia-ml-py` (necessário para `torch.cuda.utilization()` funcionar —
  sem ele a chamada levanta `RuntimeError`, confirmado neste ambiente antes de instalar).

### Corrigido / Validado nesta sessão
- **Observability executada de verdade**: Docker Desktop estava instalado mas o engine não
  estava rodando — iniciado nesta sessão. `docker compose up` direto na pasta do repo (dentro
  do Google Drive Desktop) falhou (bind-mount tratando o arquivo `prometheus.yml` como
  diretório — filesystem virtual do Drive não suporta bind mount de arquivo como um
  filesystem local). Contornado copiando os arquivos de config para uma pasta de staging
  local (`C:\tmp\rtml-observability`, fora do Drive) e subindo o compose de lá. Prometheus
  confirmado raspando `/metrics` real do serviço (`health: "up"` via API), Grafana com
  datasource e dashboard provisionados confirmados via API — evidência JSON salva em
  `observability/evidence/`.

### Documentado (ainda não implementado)
- Triton/vLLM reais (M4/M5) permanecem fora de escopo desta sessão — mock local inalterado.
- Multi-GPU, NCCL, TensorRT, quantization, voz/WebRTC/SIP: fora de escopo (V2/V3/NON-GOALS).

## [0.1.0] - 2026-10-07

### Adicionado
- Serviço de inferência assíncrono (FastAPI + WebSocket) servindo um modelo PyTorch real (`TinyCNN`), com device real detectado em runtime (`cuda` confirmado com NVIDIA RTX 3060 Ti, ou `cpu` como fallback).
- Batching dinâmico (`inference_service/batching.py`): agrupa requisições concorrentes em um único forward pass, respeitando `max_batch_size` e `max_wait_ms`.
- Cache LRU de resultados por hash do input (`inference_service/cache.py`).
- Backpressure real por profundidade de fila, retornando HTTP 429 quando saturado.
- Métricas Prometheus reais (`inference_service/metrics.py`, endpoint `/metrics`): latência, throughput por outcome, cache hit/miss, tamanho de batch, profundidade de fila, forward passes.
- Clientes HTTP reais para Triton Inference Server (protocolo KServe v2) e vLLM (API OpenAI-compatible) em `adapters/triton_vllm/`.
- Stack de observabilidade (`observability/`): `docker-compose.yml` com Prometheus + Grafana, dashboard provisionado via JSON.
- Load test real (`loadtest/load_test.py`): gera tráfego HTTP concorrente real e mede latência P50/P95/P99 e throughput medidos (não estimados).
- Suite de 23 testes automatizados cobrindo cache, batching (incluindo a prova de que N requisições concorrentes geram menos de N forward passes), backpressure, adapters (contra mock local) e integração HTTP/WebSocket end-to-end.

### Documentado (não implementado nesta versão)
- Nenhum servidor Triton Inference Server ou vLLM real foi implantado/testado nesta sessão — ver README "Status".
- Stack Prometheus/Grafana via Docker Compose não foi executada nesta sessão (configuração real, não validada em runtime Docker aqui).
