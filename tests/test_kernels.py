import pytest
import torch

from course_vllm.kernels import KernelUnavailable, assert_close, load_cuda_extension


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
