from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from course_vllm.model.qwen3_torch import Qwen3ForCausalLM


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/wangqi/huggingface/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_ids = tokenizer.encode(args.prompt, add_special_tokens=False, return_tensors="pt").to(device)

    course_model = Qwen3ForCausalLM.from_pretrained(args.model, device=device, dtype=dtype)
    with torch.inference_mode():
        course_logits = course_model(input_ids)[:, -1].float().cpu()
    del course_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    hf_model.eval()
    with torch.inference_mode():
        hf_logits = hf_model(input_ids=input_ids).logits[:, -1].float().cpu()

    diff = (course_logits - hf_logits).abs()
    print(f"max_abs_diff={diff.max().item():.6f}")
    print(f"mean_abs_diff={diff.mean().item():.6f}")
    print(f"course_top5={torch.topk(course_logits[0], 5).indices.tolist()}")
    print(f"hf_top5={torch.topk(hf_logits[0], 5).indices.tolist()}")


if __name__ == "__main__":
    main()
