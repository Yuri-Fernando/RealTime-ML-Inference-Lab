#!/usr/bin/env bash
# Sobe um servidor vLLM (OpenAI-compatible) via Docker com GPU, a partir do WSL
# ou do Git Bash, e espera ele responder em /v1/models.
#
#   bash scripts/run_vllm_wsl.sh [modelo] [porta]
#
# Depois, aponte o serviço de geração para ele:
#   RTML_BACKEND=vllm RTML_VLLM_BASE_URL=http://127.0.0.1:8001 \
#   RTML_VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
#   python -m uvicorn inference_service.llm_app:app --port 8100
#
# E rode o MESMO load test do repositório:
#   python loadtest/load_test.py --scenario llm --base-url http://127.0.0.1:8100 \
#          --sweep 1,2,4,8,16 --requests-per-level 24 --max-new-tokens 32
#
# STATUS: script não executado até o fim neste ambiente (o pull da imagem excedeu
# o tempo limite da tentativa; ver README). Nenhum benchmark de vLLM real foi
# gravado por causa disso.
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${2:-8001}"
NAME="rtml-vllm"
HF_CACHE="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

command -v docker >/dev/null || { echo "docker não encontrado"; exit 2; }

# 1) a GPU precisa estar visível para containers (--gpus all)
if ! docker run --rm --gpus all --entrypoint nvidia-smi "${VLLM_IMAGE:-vllm/vllm-openai:latest}" -L >/dev/null 2>&1; then
  echo "AVISO: não consegui validar --gpus all com a imagem do vLLM (talvez ainda não baixada)." >&2
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc=host \
  -p "${PORT}:8000" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "${VLLM_IMAGE:-vllm/vllm-openai:latest}" \
  --model "$MODEL" --dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.55

echo "esperando o vLLM ficar pronto em http://127.0.0.1:${PORT}/v1/models ..."
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "pronto após ~$((i * 5)) s"; exit 0
  fi
  sleep 5
done
echo "vLLM não respondeu em 15 min; logs:" >&2
docker logs --tail 40 "$NAME" >&2 || true
exit 1
