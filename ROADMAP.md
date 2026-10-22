# Roadmap

Sequenciamento conforme a auditoria de portfólio (seção "7) ML SYSTEMS / GPU / REAL-TIME INFERENCE")
e alinhado com os milestones M0-M7 do design doc (`03_RealTime-ML-Inference-Lab.md` seção 32).

## MVP (sessão 2026-10-07 — v0.1.0) ✅

- [x] M0/M1: Serviço assíncrono FastAPI + WebSocket servindo modelo PyTorch real (GPU/CPU conforme disponibilidade).
- [x] M3: Batching dinâmico real (testado: concorrência → menos forward passes).
- [x] Cache de resultados (hash do input, LRU).
- [x] M2 (parcial): Backpressure real (HTTP 429 sob fila saturada) — ainda inline no batcher nesta versão.
- [x] M6 (parcial): Métricas Prometheus reais + dashboard Grafana provisionado (compose não executado nesta sessão).
- [x] Adapters HTTP reais para Triton (KServe v2) e vLLM (OpenAI-compatible), testados contra mock local.
- [x] M7 (parcial): Load test real com latência P50/P95/P99 medida (um único nível de concorrência).
- [x] 23 testes automatizados, 100% passando.

## Aprofundamento (sessão 2026-10-08 — v0.2.0) ✅

- [x] **M2 — Admission Control**: extraído para `inference_service/admission.py`, decisão pura
  testável isoladamente (`tests/test_admission.py`, 7 testes), separada do `DynamicBatcher`.
- [x] **Cancellation real** (seção 13 do design doc, hipótese H5 do master spec): request
  cancelado antes do forward pass é provadamente excluído do batch (`tests/test_cancellation.py`,
  6 testes) — via `POST /cancel/{request_id}` e via desconexão real de WebSocket.
- [x] **M6 — GPU telemetry real** (seção 14): `torch.cuda.memory_allocated/reserved` +
  `torch.cuda.utilization()` (via `nvidia-ml-py`, instalado nesta sessão), lidos a cada
  scrape, expostos em `/metrics`.
- [x] **M6 — Prometheus/Grafana executados de verdade**: Docker Desktop iniciado, stack rodada
  via `docker compose up` (staging local necessário por limitação de bind-mount em pasta
  Google Drive), scrape confirmado `health: up`, dashboard confirmado provisionado — evidência
  em `observability/evidence/`.
- [x] **M7 — Concurrency sweep + capacity planning real**: `loadtest/load_test.py --sweep`
  roda 1/2/4/8/16/32/64 de concorrência sequencialmente, relatório em `loadtest/reports/`,
  ponto de saturação medido (concorrência = 8 neste hardware/modelo) em vez de estimado.
- [x] Suite de testes: 23 → 40 testes, 100% passando.

## V0.3 (sessão 2026-10-21 — v0.3.0) ✅ parcial

- [x] **Backends plugáveis** (`inference_service/backends/`): protocolo `InferenceBackend`/`TextBackend`,
  `TinyCNNBackend` (o `ModelBackend` histórico, intacto), `HuggingFaceBackend` e `VLLMBackend`,
  selecionáveis por `RTML_BACKEND` (`tinycnn` | `hf` | `vllm`).
- [x] **Serviço de geração** (`inference_service/llm_app.py` + `llm_batching.py`): `POST /generate`,
  batching dinâmico de geração (padding à esquerda, orçamento de tokens por sequência), admissão
  (429), cancelamento real (fila, durante a geração e desconexão do cliente) e métricas `rtml_llm_*`.
- [x] **Quantização real no HuggingFaceBackend**: FP32 / FP16 / BF16 / INT8 (LLM.int8) / NF4 (bitsandbytes),
  medidos com o mesmo load test (`loadtest/load_test.py --scenario llm`, concurrency sweep) —
  resultados em `benchmarks/results/`, tabela no README.
- [x] **Prometheus/Grafana** do caminho LLM: job de scrape na porta 8100 e
  `observability/grafana/dashboards/llm-dashboard.json`.
- [x] `VLLMBackend` completo, testado contra o servidor mock OpenAI-compatible; `docker-compose.vllm.yml` e
  `scripts/run_vllm_wsl.sh` documentados.
- [ ] **vLLM real — BLOCKED_BY_CONFIGURATION**: o `docker pull vllm/vllm-openai` concluiu 8 de 40 camadas
  no limite de 15 min da tentativa; a imagem não ficou disponível, então nenhum benchmark de vLLM
  real foi executado. Próximo passo: baixar a imagem com tempo/rede suficientes e rodar
  `scripts/run_vllm_wsl.sh` + `load_test.py --scenario llm` contra ele.
- [ ] TTFT por token com streaming (hoje: espera em fila + prefill do lote; vLLM sem TTFT).

## M4/M5 (não tentado — fora de escopo das duas sessões até aqui)

- [ ] **vLLM real**: `VLLMBackend` + adapter HTTP existem e foram testados só contra mock local;
  a tentativa de subir o vLLM real via Docker (V0.3) ficou BLOCKED_BY_CONFIGURATION (ver V0.3 acima).
- [ ] **Triton real**: idem, `adapters/triton_vllm/triton_client.py` — nenhum NVIDIA Triton
  Inference Server real foi implantado. Ambos exigem infraestrutura/modelo real fora do
  escopo orçado para as sessões realizadas até aqui.

## V2 (depois)

- [ ] **TensorRT**: conversão do modelo para engine TensorRT, comparação de latência contra o backend PyTorch puro.
- [x] ~~**Quantization**: INT8/FP16 post-training, medir impacto real~~ — feito na V0.3 para o `HuggingFaceBackend`
  (FP32/FP16/BF16/INT8/NF4 via bitsandbytes). Ainda aberto: quantização do TinyCNN e QAT.
- [ ] **CUDA Graphs**: capturar o forward pass em um CUDA Graph para eliminar overhead de kernel launch/sincronização por batch — ataca diretamente o achado de latência documentado no README (overhead de seleção de algoritmo cuDNN por batch de tamanho variável).
- [ ] **Ray / Kubernetes GPU**: orquestração do serviço em múltiplos workers/pods com GPU, autoscaling baseado em profundidade de fila/latência.
- [ ] Validação end-to-end real contra um servidor Triton Inference Server e/ou vLLM real (não apenas mock) — ver seção "M4/M5" acima, ainda não tentado.
- [x] ~~Validação do stack Prometheus/Grafana via Docker Compose em runtime real~~ — feito na sessão 2026-10-08, ver seção acima e README "Observability (validado)".

## V3 (depois)

- [ ] **Multi-GPU / NCCL**: particionamento de modelo ou replicação entre múltiplas GPUs, comunicação via NCCL, medir throughput agregado real.
- [ ] Suporte a streaming de áudio em tempo real (WebRTC/RTP/SIP) como caso de uso adicional de baixa latência, conforme mencionado na auditoria original.

## Fora de escopo (por ora)

- Treinamento do modelo servido (o objetivo deste repo é a infraestrutura de serving, não a tarefa de ML em si — ver README "Modelagem").
- Multi-tenancy / autenticação de API (não mencionado na auditoria original como requisito do MVP).
