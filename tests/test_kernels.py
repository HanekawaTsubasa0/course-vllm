import pytest
import torch

from course_vllm.kernels import (
    KernelUnavailable,
    assert_close,
    load_cuda_extension,
    triton_matmul,
    triton_rms_norm,
    triton_rope,
    triton_softmax,
)
from course_vllm.model.qwen3_torch import rotate_half


def test_vector_add_cuda_kernel_matches_torch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    try:
        module = load_cuda_extension("course_vllm_vector_add", ("kernels/vector_add.cu",))
    except KernelUnavailable as exc:
        pytest.skip(str(exc))
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    assert_close("vector_add", module.vector_add(a, b), a + b)


def test_triton_softmax_matches_torch():
    _require_cuda()
    x = torch.randn(5, 17, device="cuda")
    actual = _run_or_skip(triton_softmax, x)
    assert_close("triton_softmax", actual, torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-5)


def test_triton_rms_norm_matches_torch():
    _require_cuda()
    x = torch.randn(7, 32, device="cuda")
    weight = torch.randn(32, device="cuda")
    actual = _run_or_skip(triton_rms_norm, x, weight, eps=1e-6)
    expected = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6) * weight
    assert_close("triton_rms_norm", actual, expected, atol=1e-5, rtol=1e-5)


def test_triton_rope_matches_qwen3_rotate_half():
    _require_cuda()
    x = torch.randn(6, 32, device="cuda")
    angles = torch.randn_like(x)
    cos = angles.cos()
    sin = angles.sin()
    actual = _run_or_skip(triton_rope, x, cos, sin)
    expected = x * cos + rotate_half(x) * sin
    assert_close("triton_rope", actual, expected, atol=1e-5, rtol=1e-5)


def test_triton_matmul_matches_torch():
    _require_cuda()
    a = torch.randn(19, 33, device="cuda")
    b = torch.randn(33, 11, device="cuda")
    actual = _run_or_skip(triton_matmul, a, b)
    assert_close("triton_matmul", actual, a @ b, atol=1e-3, rtol=1e-3)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _run_or_skip(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KernelUnavailable as exc:
        pytest.skip(str(exc))
