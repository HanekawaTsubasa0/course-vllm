#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-profiles/ncu_kernels}"
KERNEL_SCENARIO="${KERNEL_SCENARIO:-paged_attention}"
KERNEL_NAME="${KERNEL_NAME:-}"
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

case "$KERNEL_SCENARIO" in
  rmsnorm)
    TARGETS=(tests/test_kernels.py::test_cuda_rms_norm_matches_torch)
    ;;
  rope)
    TARGETS=(tests/test_kernels.py::test_cuda_rope_matches_qwen3_rotate_half)
    ;;
  softmax)
    TARGETS=(tests/test_kernels.py::test_cuda_softmax_matches_torch)
    ;;
  matmul)
    TARGETS=(tests/test_kernels.py::test_cuda_matmul_tiled_matches_torch)
    ;;
  dense_attention)
    TARGETS=(tests/test_attention.py::test_cuda_dense_attention_decode_matches_reference)
    ;;
  paged_attention)
    TARGETS=(tests/test_attention.py::test_cuda_paged_attention_decode_matches_dense_attention)
    ;;
  all)
    TARGETS=(tests/test_kernels.py tests/test_attention.py)
    ;;
  *)
    echo "unknown KERNEL_SCENARIO=$KERNEL_SCENARIO" >&2
    echo "choose one of: rmsnorm rope softmax matmul dense_attention paged_attention all" >&2
    exit 2
    ;;
esac

NCU_ARGS=(
  --set full
  --force-overwrite
  --export "$OUT"
)
if [[ -n "$KERNEL_NAME" ]]; then
  NCU_ARGS+=(--kernel-name "$KERNEL_NAME")
fi

echo "ncu scenario=$KERNEL_SCENARIO kernel_name=${KERNEL_NAME:-<all>} export=$OUT"
"$NCU_BIN" \
  "${NCU_ARGS[@]}" \
  python -m pytest -q "${TARGETS[@]}" -rs
