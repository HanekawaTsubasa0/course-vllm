from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabTarget:
    lab: str
    files: tuple[str, ...]
    focus: str


TARGETS = (
    LabTarget("lab03", ("kernels/vector_add.cu",), "vector add kernel body"),
    LabTarget(
        "lab04",
        ("kernels/course_ops.cu", "course_vllm/model/qwen3_torch.py"),
        "RMSNorm/RoPE kernels and dispatch",
    ),
    LabTarget(
        "lab05",
        ("kernels/course_ops.cu", "course_vllm/model/ops.py"),
        "naive/tiled matmul and CourseLinear dispatch",
    ),
    LabTarget(
        "lab06",
        ("kernels/course_ops.cu", "course_vllm/engine/sampler.py"),
        "stable softmax and sampling dispatch",
    ),
    LabTarget(
        "lab07",
        ("kernels/course_ops.cu", "course_vllm/model/attention.py", "course_vllm/model/ops.py"),
        "attention CUDA path",
    ),
    LabTarget("lab08", ("course_vllm/engine/kv_cache.py",), "continuous KV cache"),
    LabTarget(
        "lab09",
        ("course_vllm/engine/engine.py", "course_vllm/engine/request.py"),
        "request lifecycle and streaming loop",
    ),
    LabTarget(
        "lab10",
        ("course_vllm/engine/block_manager.py", "course_vllm/engine/paged_kv_cache.py"),
        "paged KV cache and block table",
    ),
    LabTarget(
        "lab11",
        ("course_vllm/engine/scheduler.py", "course_vllm/server/batching.py"),
        "continuous batching and preemption",
    ),
    LabTarget(
        "lab12",
        ("course_vllm/model/qwen3_continuous_backend.py", "course_vllm/server/batching.py"),
        "pinned memory, streams, and admission control",
    ),
)


def main() -> None:
    print("Student branch starter-code target preview")
    print("=" * 43)
    for target in TARGETS:
        print(f"{target.lab}: {target.focus}")
        for path in target.files:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
