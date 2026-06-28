#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${1:-/tmp/course-vllm-clean-smoke}"
PORT="${PORT:-18180}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
REMOTE_URL="${REMOTE_URL:-}"
export PORT

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN=""
  for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v "$candidate")"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Could not find Python 3.10, 3.11, or 3.12. Set PYTHON=/path/to/python and retry." >&2
  exit 2
fi

case "$WORKDIR" in
  /tmp/course-vllm-*|/tmp/*/course-vllm-*) ;;
  *)
    echo "Refusing to delete non-temporary smoke directory: $WORKDIR" >&2
    echo "Use a path under /tmp, for example /tmp/course-vllm-clean-smoke." >&2
    exit 2
    ;;
esac

if [[ -z "$REMOTE_URL" && -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  echo "Refusing clean-clone smoke from a dirty worktree." >&2
  echo "Commit or stash changes first, then rerun this script." >&2
  exit 2
fi

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
if [[ -n "$REMOTE_URL" ]]; then
  git clone "$REMOTE_URL" "$WORKDIR/repo"
else
  git clone "$REPO_ROOT" "$WORKDIR/repo"
fi
cd "$WORKDIR/repo"

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

pytest -q -rs --ignore=tests/test_kernels.py --ignore=tests/test_attention.py
python -m course_vllm.benchmarks.grader week01
python -m course_vllm.benchmarks.grader week02
python -m course_vllm.benchmarks.grader week11
python -m course_vllm.benchmarks.grader week12

python -m course_vllm.server.api \
  --model "$MODEL" \
  --backend paged \
  --stage week01 \
  --kernel-impl auto \
  --dtype bfloat16 \
  --port "$PORT" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

python - <<'PY'
import os
import time

import httpx

port = int(os.environ.get("PORT", "18180"))
base = f"http://127.0.0.1:{port}"
for _ in range(120):
    try:
        r = httpx.get(f"{base}/health", timeout=1)
        if r.status_code == 200:
            break
    except httpx.HTTPError:
        pass
    time.sleep(0.5)
else:
    raise SystemExit("server did not become ready")

payload = {"prompt": "Hello", "stream": False, "sampling_params": {"temperature": 0, "max_tokens": 4}}
r = httpx.post(f"{base}/generate", json=payload, timeout=30)
r.raise_for_status()
body = r.json()
assert "text" in body
print("HTTP smoke ok:", body["text"][:80])
PY

echo "Clean-clone smoke completed in $WORKDIR/repo"
