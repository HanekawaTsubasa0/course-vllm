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
