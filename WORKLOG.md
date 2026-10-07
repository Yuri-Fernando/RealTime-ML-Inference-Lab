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
