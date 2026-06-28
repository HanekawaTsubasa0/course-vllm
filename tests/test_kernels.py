import os

import pytest
import torch

from course_vllm.kernels import (
    assert_close,
    benchmark_cuda,
    cuda_matmul,
    cuda_matmul_tiled,
    cuda_rms_norm,
    cuda_rope,
    cuda_softmax,
    KernelUnavailable,
    load_cuda_extension,
)
from course_vllm.model.qwen3_torch import rotate_half


def test_vector_add_cuda_kernel_matches_torch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    try:
        module = load_cuda_extension("course_vllm_vector_add", ("kernels/vector_add.cu",))
    except KernelUnavailable as exc:
        _skip_or_raise_kernel_unavailable(exc)
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    assert_close("vector_add", module.vector_add(a, b), a + b)


def test_cuda_softmax_matches_torch():
    _require_cuda()
    x = torch.randn(5, 17, device="cuda")
    try:
        actual = cuda_softmax(x)
    except KernelUnavailable as exc:
        _skip_or_raise_kernel_unavailable(exc)
    assert_close("cuda_softmax", actual, torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-5)


def test_cuda_rms_norm_matches_torch():
    _require_cuda()
    x = torch.randn(7, 32, device="cuda")
    weight = torch.randn(32, device="cuda")
    try:
        actual = cuda_rms_norm(x, weight, eps=1e-6)
    except KernelUnavailable as exc:
        _skip_or_raise_kernel_unavailable(exc)
    expected = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6) * weight
    assert_close("cuda_rms_norm", actual, expected, atol=1e-5, rtol=1e-5)


def test_cuda_rope_matches_qwen3_rotate_half():
    _require_cuda()
    x = torch.randn(6, 32, device="cuda")
    angles = torch.randn_like(x)
    cos = angles.cos()
    sin = angles.sin()
    try:
        actual = cuda_rope(x, cos, sin)
    except KernelUnavailable as exc:
        _skip_or_raise_kernel_unavailable(exc)
    expected = x * cos + rotate_half(x) * sin
    assert_close("cuda_rope", actual, expected, atol=1e-5, rtol=1e-5)


def test_cuda_matmul_matches_torch():
    _require_cuda()
    a = torch.randn(19, 33, device="cuda")
    b = torch.randn(33, 11, device="cuda")
    try:
        actual = cuda_matmul(a, b)
    except KernelUnavailable as exc:
        _skip_or_raise_kernel_unavailable(exc)
    assert_close("cuda_matmul", actual, a @ b, atol=1e-3, rtol=1e-3)


def test_cuda_matmul_tiled_matches_torch():
    _require_cuda()
    a = torch.randn(31, 65, device="cuda")
    b = torch.randn(65, 17, device="cuda")
    try:
        actual = cuda_matmul_tiled(a, b)
    except KernelUnavailable as exc:
        _skip_or_raise_kernel_unavailable(exc)
    assert_close("cuda_matmul_tiled", actual, a @ b, atol=1e-3, rtol=1e-3)


def test_cuda_matmul_not_much_slower_than_torch_on_small_teaching_case():
    _require_cuda()
    a = torch.randn(64, 128, device="cuda")
    b = torch.randn(128, 64, device="cuda")
    try:
        cuda_matmul(a, b)
    except KernelUnavailable as exc:
        _skip_or_raise_kernel_unavailable(exc)
    cuda_ms = benchmark_cuda(lambda: cuda_matmul(a, b), warmup=3, repeat=10)
    torch_ms = benchmark_cuda(lambda: a @ b, warmup=3, repeat=10)
    assert cuda_ms < torch_ms * 20 + 0.05


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _skip_or_raise_kernel_unavailable(exc: KernelUnavailable) -> None:
    if os.environ.get("COURSE_VLLM_STRICT_CUDA") == "1":
        raise exc
    pytest.skip(f"CUDA kernel unavailable: {exc}")
