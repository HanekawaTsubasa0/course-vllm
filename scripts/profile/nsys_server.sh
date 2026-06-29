#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-18080}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
BACKEND="${BACKEND:-course}"
KV_MODE="${KV_MODE:-paged}"
DTYPE="${DTYPE:-bfloat16}"
MAX_TOKENS="${MAX_TOKENS:-32}"
OUT="${OUT:-profiles/nsys_server}"

mkdir -p "$(dirname "$OUT")"

NSYS_BIN="${NSYS_BIN:-$(command -v nsys || true)}"
if [[ -z "$NSYS_BIN" ]]; then
  for candidate in /usr/local/cuda-12.8/bin/nsys /usr/local/cuda/bin/nsys /usr/local/bin/nsys; do
    if [[ -x "$candidate" ]]; then
      NSYS_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$NSYS_BIN" ]]; then
  echo "nsys not found; set NSYS_BIN=/path/to/nsys" >&2
  exit 127
fi

"$NSYS_BIN" profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --force-overwrite=true \
  --output="$OUT" \
python -m course_vllm.server.api \
    --model "$MODEL" \
    --backend "$BACKEND" \
    --kv-mode "$KV_MODE" \
    --dtype "$DTYPE" \
    --stage week02 \
    --port "$PORT" &

SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

python - <<PY
import sys
import time

import httpx

url = "http://127.0.0.1:${PORT}/health"
deadline = time.time() + float("${STARTUP_TIMEOUT:-120}")
last_error = None
while time.time() < deadline:
    try:
        response = httpx.get(url, timeout=1.0)
        if response.status_code == 200:
            print(f"server ready: {url}")
            sys.exit(0)
    except Exception as exc:
        last_error = exc
    time.sleep(1)
raise SystemExit(f"server did not become ready before timeout; last_error={last_error}")
PY

python -m course_vllm.benchmarks.bench_server \
  --url "http://127.0.0.1:${PORT}/generate" \
  --num-requests 4 \
  --concurrency 1 \
  --max-tokens "$MAX_TOKENS"
