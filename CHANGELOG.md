# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versionamento [SemVer](https://semver.org/lang/pt-BR/).

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
