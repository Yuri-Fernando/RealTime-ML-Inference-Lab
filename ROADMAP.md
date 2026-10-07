# Roadmap

Sequenciamento conforme a auditoria de portfólio (seção "7) ML SYSTEMS / GPU / REAL-TIME INFERENCE").

## MVP (esta sessão — v0.1.0) ✅

- [x] Serviço assíncrono FastAPI + WebSocket servindo modelo PyTorch real (GPU/CPU conforme disponibilidade).
- [x] Batching dinâmico real (testado: concorrência → menos forward passes).
- [x] Cache de resultados (hash do input, LRU).
- [x] Backpressure real (HTTP 429 sob fila saturada).
- [x] Métricas Prometheus reais + dashboard Grafana provisionado (compose não executado nesta sessão).
- [x] Adapters HTTP reais para Triton (KServe v2) e vLLM (OpenAI-compatible), testados contra mock local.
- [x] Load test real com latência P50/P95/P99 medida.
- [x] 23 testes automatizados, 100% passando.

## V2 (depois)

- [ ] **TensorRT**: conversão do modelo para engine TensorRT, comparação de latência contra o backend PyTorch puro.
- [ ] **Quantization**: INT8/FP16 quantization-aware ou post-training, medir impacto real em latência e (quando aplicável) acurácia.
- [ ] **CUDA Graphs**: capturar o forward pass em um CUDA Graph para eliminar overhead de kernel launch/sincronização por batch — ataca diretamente o achado de latência documentado no README (overhead de seleção de algoritmo cuDNN por batch de tamanho variável).
- [ ] **Ray / Kubernetes GPU**: orquestração do serviço em múltiplos workers/pods com GPU, autoscaling baseado em profundidade de fila/latência.
- [ ] Validação end-to-end real contra um servidor Triton Inference Server e/ou vLLM real (não apenas mock) — depende de infraestrutura/acesso não disponível na sessão do MVP.
- [ ] Validação do stack Prometheus/Grafana via Docker Compose em runtime real (não executado no MVP).

## V3 (depois)

- [ ] **Multi-GPU / NCCL**: particionamento de modelo ou replicação entre múltiplas GPUs, comunicação via NCCL, medir throughput agregado real.
- [ ] Suporte a streaming de áudio em tempo real (WebRTC/RTP/SIP) como caso de uso adicional de baixa latência, conforme mencionado na auditoria original.

## Fora de escopo (por ora)

- Treinamento do modelo servido (o objetivo deste repo é a infraestrutura de serving, não a tarefa de ML em si — ver README "Modelagem").
- Multi-tenancy / autenticação de API (não mencionado na auditoria original como requisito do MVP).
