from __future__ import annotations

import argparse

from course_vllm import Engine, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/wangqi/huggingface/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Hello, introduce yourself briefly.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--chat", action="store_true")
    args = parser.parse_args()

    engine = Engine(args.model, dtype=args.dtype)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
    if args.chat:
        result = engine.chat([{"role": "user", "content": args.prompt}], params)
    else:
        result = engine.generate(args.prompt, params)
    print(result["text"])


if __name__ == "__main__":
    main()
