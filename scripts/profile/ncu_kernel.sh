#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-profiles/ncu_kernels}"
mkdir -p "$(dirname "$OUT")"

NCU_BIN="${NCU_BIN:-$(command -v ncu || true)}"
if [[ -z "$NCU_BIN" ]]; then
  for candidate in /usr/local/cuda-12.8/bin/ncu /usr/local/cuda/bin/ncu /usr/local/NVIDIA-Nsight-Compute-2025.3/ncu; do
    if [[ -x "$candidate" ]]; then
      NCU_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$NCU_BIN" ]]; then
  echo "ncu not found; set NCU_BIN=/path/to/ncu" >&2
  exit 127
fi

"$NCU_BIN" \
  --set full \
  --force-overwrite \
  --export "$OUT" \
  python -m pytest -q tests/test_kernels.py tests/test_attention.py -rs
