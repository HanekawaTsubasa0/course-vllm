from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from course_vllm.model.qwen3_backend import Qwen3PagedBackend, Qwen3TorchBackend
from course_vllm.model.types import parse_dtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/wangqi/huggingface/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--token-ids", default="21806,14582,15846")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--backend", default="course", choices=["course", "paged"])
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    decode_token_ids = [int(item) for item in args.token_ids.split(",") if item]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)

    backend_cls = Qwen3PagedBackend if args.backend == "paged" else Qwen3TorchBackend
    course = backend_cls(args.model, device=str(device), dtype=args.dtype)
    course_out = course.prefill(prompt_ids)
    course_logits = [course_out.logits.float().cpu()]
    course_cache = course_out.past_key_values
    for token_id in decode_token_ids:
        course_out = course.decode_step(token_id, course_cache)
        course_logits.append(course_out.logits.float().cpu())
        course_cache = course_out.past_key_values
    del course
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    hf_model.eval()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        hf_out = hf_model(input_ids=input_ids, use_cache=True)
    hf_logits = [hf_out.logits[0, -1].float().cpu()]
    hf_cache = hf_out.past_key_values
    for token_id in decode_token_ids:
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
        with torch.inference_mode():
            hf_out = hf_model(input_ids=input_ids, past_key_values=hf_cache, use_cache=True)
        hf_logits.append(hf_out.logits[0, -1].float().cpu())
        hf_cache = hf_out.past_key_values

    for step, (course_step_logits, hf_step_logits) in enumerate(zip(course_logits, hf_logits)):
        diff = (course_step_logits - hf_step_logits).abs()
        print(
            f"step={step} max_abs_diff={diff.max().item():.6f} "
            f"mean_abs_diff={diff.mean().item():.6f}"
        )
        print(f"  course_top5={torch.topk(course_step_logits, 5).indices.tolist()}")
        print(f"  hf_top5={torch.topk(hf_step_logits, 5).indices.tolist()}")


if __name__ == "__main__":
    main()
