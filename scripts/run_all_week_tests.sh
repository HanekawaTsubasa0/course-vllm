#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi

STAGES=(
  week01
  week02
  week03
  week04
  week05
  week06
  week07
  week08
  week09
  week10
  week11
  week12
  week13
  week15
  cuda_smoke
)

echo "course-vllm staged validation"
echo "root: $ROOT"
echo "python: $PYTHON"
echo

if ! "$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; strict all-week validation requires CUDA.")
print("cuda_available: True")
print(f"cuda_device_count: {torch.cuda.device_count()}")
print(f"cuda_device_0: {torch.cuda.get_device_name(0)}")
print(f"torch: {torch.__version__}")
print(f"torch_cuda: {torch.version.cuda}")
PY
then
  echo "ERROR: CUDA preflight failed. Run this script in a GPU-visible environment." >&2
  exit 1
fi

export COURSE_VLLM_STRICT_CUDA=1
echo "strict_cuda: COURSE_VLLM_STRICT_CUDA=1"
echo

PASS=()
FAIL=()

for stage in "${STAGES[@]}"; do
  echo "=============================="
  echo "RUN $stage"
  echo "=============================="
  if "$PYTHON" -m course_vllm.benchmarks.grader "$stage"; then
    PASS+=("$stage")
    echo "PASS $stage"
  else
    FAIL+=("$stage")
    echo "FAIL $stage"
  fi
  echo
done

echo "=============================="
echo "SUMMARY"
echo "=============================="
echo "passed (${#PASS[@]}): ${PASS[*]:-none}"
echo "failed (${#FAIL[@]}): ${FAIL[*]:-none}"
echo "week14: deferred, no grader"
echo "week16: final review uses the combined staged tests above"

if (( ${#FAIL[@]} > 0 )); then
  exit 1
fi
