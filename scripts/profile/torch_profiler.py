from __future__ import annotations

import argparse
from pathlib import Path

import torch

from course_vllm import Engine, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--backend", default="paged", choices=["hf", "course", "paged"])
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--out", default="profiles/torch_profiler")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = Engine(args.model, backend=args.backend, stage="week09", kernel_impl="auto")
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out_dir)),
    ) as prof:
        engine.generate(args.prompt, params)
        prof.step()
    print(prof.key_averages().table(sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total", row_limit=20))


if __name__ == "__main__":
    main()
