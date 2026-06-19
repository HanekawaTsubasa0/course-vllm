from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


class KernelUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=8)
def load_cuda_extension(name: str, sources: tuple[str, ...]):
    if not torch.cuda.is_available():
        raise KernelUnavailable("CUDA is not available")
    root = Path(__file__).resolve().parents[2]
    try:
        return load(
            name=name,
            sources=[str(root / source) for source in sources],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        raise KernelUnavailable(str(exc)) from exc


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-6) -> None:
    diff = (actual.float() - expected.float()).abs()
    if not torch.allclose(actual, expected, atol=atol, rtol=0):
        raise AssertionError(f"{name}: max_abs_diff={diff.max().item():.6f}")


def benchmark_cuda(fn, *, warmup: int = 10, repeat: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat
