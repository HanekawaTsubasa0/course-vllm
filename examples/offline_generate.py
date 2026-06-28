from __future__ import annotations

import argparse

from course_vllm import Engine, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Hello, introduce yourself briefly.")
    parser.add_argument("--prompts", default="")
    parser.add_argument("--max-tokens", type=int, default=None, help="maximum generated tokens; omit for no explicit limit")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--backend", default="hf", choices=["hf", "course", "paged"])
    parser.add_argument("--stage", default=None, help="course stage such as week04 or 4")
    parser.add_argument("--kernel-impl", default="torch", choices=["torch", "auto", "cuda"])
    parser.add_argument("--pinned-memory", action="store_true")
    parser.add_argument("--transfer-stream", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--cache-aware-scheduling", action="store_true")
    parser.add_argument("--chat", action="store_true")
    args = parser.parse_args()

    engine = Engine(
        args.model,
        dtype=args.dtype,
        backend=args.backend,
        stage=args.stage,
        kernel_impl=args.kernel_impl,
        use_pinned_memory=args.pinned_memory,
        use_transfer_stream=args.transfer_stream,
    )
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
    if args.prompts:
        prompts = [prompt for prompt in args.prompts.split("|") if prompt]
        results = engine.generate_batch(
            prompts,
            params,
            enable_chunked_prefill=args.enable_chunked_prefill,
            cache_aware_scheduling=args.cache_aware_scheduling,
        )
        for prompt, result in zip(prompts, results):
            print(f"prompt={prompt!r}")
            print(result["text"])
        return
    if args.chat:
        result = engine.chat([{"role": "user", "content": args.prompt}], params)
    else:
        result = engine.generate(args.prompt, params)
    print(result["text"])


if __name__ == "__main__":
    main()
