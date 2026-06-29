#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"
DTYPE="${DTYPE:-bfloat16}"
KERNEL_IMPL="${KERNEL_IMPL:-cuda}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
BATCH_WAIT_MS="${BATCH_WAIT_MS:-2}"
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-128}"
MAX_PROMPT_CHARS="${MAX_PROMPT_CHARS:-8192}"

echo "Starting final course-vllm server"
echo "root: $ROOT"
echo "python: $PYTHON"
echo "model: $MODEL"
echo "kernel_impl: $KERNEL_IMPL"
echo "url: http://$HOST:$PORT"
echo

if [[ "$KERNEL_IMPL" != "cuda" ]]; then
  echo "ERROR: final server script requires KERNEL_IMPL=cuda for strict CUDA execution." >&2
  echo "Set KERNEL_IMPL=cuda or edit the script only for non-final demos." >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; final server requires CUDA.")
print("cuda_available: True")
print(f"cuda_device_count: {torch.cuda.device_count()}")
print(f"cuda_device_0: {torch.cuda.get_device_name(0)}")
print(f"torch: {torch.__version__}")
print(f"torch_cuda: {torch.version.cuda}")
PY

export COURSE_VLLM_STRICT_CUDA=1
echo "strict_cuda: COURSE_VLLM_STRICT_CUDA=1"
echo

exec "$PYTHON" -m course_vllm.server.api \
  --model "$MODEL" \
  --backend course \
  --kv-mode paged \
  --stage week16 \
  --kernel-impl "$KERNEL_IMPL" \
  --dtype "$DTYPE" \
  --max-batch-size "$MAX_BATCH_SIZE" \
  --batch-wait-ms "$BATCH_WAIT_MS" \
  --max-queue-size "$MAX_QUEUE_SIZE" \
  --max-prompt-chars "$MAX_PROMPT_CHARS" \
  --enable-chunked-prefill \
  --cache-aware-scheduling \
  --pinned-memory \
  --transfer-stream \
  --host "$HOST" \
  --port "$PORT"
