from __future__ import annotations

import argparse

from course_vllm import Engine, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/wangqi/huggingface/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Hello, introduce yourself briefly.")
    parser.add_argument("--prompts", default="")
    parser.add_argument("--max-tokens", type=int, default=None, help="maximum generated tokens; omit for no explicit limit")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--backend", default="hf", choices=["hf", "course", "paged"])
    parser.add_argument("--chat", action="store_true")
    args = parser.parse_args()

    engine = Engine(args.model, dtype=args.dtype, backend=args.backend)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
    if args.prompts:
        prompts = [prompt for prompt in args.prompts.split("|") if prompt]
        results = engine.generate_batch(prompts, params)
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
