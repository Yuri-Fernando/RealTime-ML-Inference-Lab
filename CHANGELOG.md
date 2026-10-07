# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versionamento [SemVer](https://semver.org/lang/pt-BR/).

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
