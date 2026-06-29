from __future__ import annotations

import argparse
from pathlib import Path

import torch

from course_vllm import Engine, SamplingParams


def run_workload(engine: Engine, prompt: str, max_tokens: int, workload: str) -> None:
    if workload == "prefill":
        engine.generate(prompt, SamplingParams(temperature=0.0, max_tokens=1))
    elif workload == "decode":
        engine.generate(prompt, SamplingParams(temperature=0.0, max_tokens=max_tokens))
    elif workload == "mixed":
        prompts = [prompt, f"{prompt} Explain KV cache briefly."]
        engine.generate_batch(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens))
    else:
        raise ValueError(f"unsupported workload: {workload}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--backend", default="course", choices=["reference", "course", "hf", "paged"])
    parser.add_argument("--kv-mode", default="paged", choices=["dense", "paged"])
    parser.add_argument("--workload", default="mixed", choices=["prefill", "decode", "mixed"])
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", default="profiles/torch_profiler")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = Engine(args.model, backend=args.backend, kv_mode=args.kv_mode, stage="week09", kernel_impl="auto")
    for _ in range(args.warmup):
        run_workload(engine, args.prompt, args.max_tokens, args.workload)
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
        for _ in range(args.repeat):
            run_workload(engine, args.prompt, args.max_tokens, args.workload)
            prof.step()
    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    print(f"workload={args.workload} warmup={args.warmup} repeat={args.repeat}")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))


if __name__ == "__main__":
    main()
