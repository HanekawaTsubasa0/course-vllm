from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from course_vllm.model.qwen3_backend import Qwen3PagedBackend, Qwen3TorchBackend
from course_vllm.model.qwen3_torch import Qwen3ForCausalLM
from course_vllm.model.types import parse_dtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["forward", "decode", "batch-prefill", "batch-decode"])
    parser.add_argument("--model", default="/home/wangqi/huggingface/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--token-ids", default="21806,14582,15846")
    parser.add_argument("--batch-token-ids", default="9707,0,21806|9707,0,14582")
    parser.add_argument("--decode-token-ids", default="279,25")
    parser.add_argument("--backend", default="course", choices=["course", "paged"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()
    {
        "forward": compare_forward,
        "decode": compare_decode,
        "batch-prefill": compare_batch_prefill,
        "batch-decode": compare_batch_decode,
    }[args.mode](args)


def compare_forward(args: argparse.Namespace) -> None:
    device, dtype = torch.device(args.device), parse_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_ids = tokenizer.encode(args.prompt, add_special_tokens=False, return_tensors="pt").to(device)
    course = Qwen3ForCausalLM.from_pretrained(args.model, device=device, dtype=dtype)
    with torch.inference_mode():
        course_logits = course(input_ids)[:, -1].float().cpu()
    del course
    empty_cache(device)
    hf = hf_model(args, device, dtype)
    with torch.inference_mode():
        hf_logits = hf(input_ids=input_ids).logits[:, -1].float().cpu()
    report("course forward vs hf forward", course_logits, hf_logits)
    print(f"course_top5={torch.topk(course_logits[0], 5).indices.tolist()}")
    print(f"hf_top5={torch.topk(hf_logits[0], 5).indices.tolist()}")


def compare_decode(args: argparse.Namespace) -> None:
    device, dtype = torch.device(args.device), parse_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    decode_ids = parse_ids(args.token_ids)
    course = backend(args, device)
    out = course.prefill(prompt_ids)
    course_logits, cache = [out.logits.float().cpu()], out.past_key_values
    for token_id in decode_ids:
        out = course.decode_step(token_id, cache)
        course_logits.append(out.logits.float().cpu())
        cache = out.past_key_values
    del course
    empty_cache(device)
    hf_logits = hf_decode_logits(hf_model(args, device, dtype), prompt_ids, decode_ids, device)
    for step, (lhs, rhs) in enumerate(zip(course_logits, hf_logits)):
        report(f"step={step} course decode vs hf decode", lhs, rhs)


def compare_batch_prefill(args: argparse.Namespace) -> None:
    device, dtype = torch.device(args.device), parse_dtype(args.dtype)
    batch_ids = parse_batch_ids(args.batch_token_ids)
    batched = backend(args, device)
    course_batch = torch.stack([x.float().cpu() for x in batched.prefill_batch(batch_ids).logits])
    del batched
    empty_cache(device)
    single = backend(args, device)
    course_single = torch.stack([single.prefill(ids).logits.float().cpu() for ids in batch_ids])
    del single
    empty_cache(device)
    hf = hf_model(args, device, dtype)
    hf_single = torch.stack([hf_prefill_logits(hf, ids, device) for ids in batch_ids])
    report("course batch vs course single", course_batch, course_single)
    report("course single vs hf single", course_single, hf_single)
    if len({len(ids) for ids in batch_ids}) == 1:
        with torch.inference_mode():
            hf_batch = hf(input_ids=torch.tensor(batch_ids, dtype=torch.long, device=device), use_cache=True)
        report("hf batch vs hf single", hf_batch.logits[:, -1].float().cpu(), hf_single)


def compare_batch_decode(args: argparse.Namespace) -> None:
    device, dtype = torch.device(args.device), parse_dtype(args.dtype)
    prompt_ids, decode_ids = parse_batch_ids(args.batch_token_ids), parse_ids(args.decode_token_ids)
    if len(prompt_ids) != len(decode_ids):
        raise ValueError("number of prompts must match number of decode tokens")
    course = backend(args, device)
    prefill = course.prefill_batch(prompt_ids)
    decode = course.decode_batch(decode_ids, prefill.past_key_values)
    course_logits = torch.stack([x.float().cpu() for x in decode.logits])
    del course
    empty_cache(device)
    hf = hf_model(args, device, dtype)
    hf_single = torch.stack([hf_decode_logits(hf, ids, [tok], device)[-1] for ids, tok in zip(prompt_ids, decode_ids)])
    with torch.inference_mode():
        hf_prefill = hf(input_ids=torch.tensor(prompt_ids, dtype=torch.long, device=device), use_cache=True)
        hf_batch = hf(
            input_ids=torch.tensor([[x] for x in decode_ids], dtype=torch.long, device=device),
            past_key_values=hf_prefill.past_key_values,
            use_cache=True,
        ).logits[:, -1].float().cpu()
    report("course batch decode vs hf batch decode", course_logits, hf_batch)
    report("hf batch decode vs hf single decode", hf_batch, hf_single)


def backend(args: argparse.Namespace, device: torch.device):
    return (Qwen3PagedBackend if args.backend == "paged" else Qwen3TorchBackend)(
        args.model,
        device=str(device),
        dtype=args.dtype,
    )


def hf_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model


def hf_prefill_logits(model, token_ids: list[int], device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        return model(
            input_ids=torch.tensor([token_ids], dtype=torch.long, device=device),
            use_cache=True,
        ).logits[0, -1].float().cpu()


def hf_decode_logits(model, prompt_ids: list[int], decode_ids: list[int], device: torch.device) -> list[torch.Tensor]:
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([prompt_ids], dtype=torch.long, device=device), use_cache=True)
    logits, cache = [out.logits[0, -1].float().cpu()], out.past_key_values
    for token_id in decode_ids:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([[token_id]], dtype=torch.long, device=device),
                past_key_values=cache,
                use_cache=True,
            )
        logits.append(out.logits[0, -1].float().cpu())
        cache = out.past_key_values
    return logits


def parse_ids(value: str) -> list[int]:
    return [int(token_id) for token_id in value.split(",") if token_id]


def parse_batch_ids(value: str) -> list[list[int]]:
    return [parse_ids(row) for row in value.split("|") if row]


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def report(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    diff = (lhs - rhs).abs()
    print(f"{name}: max_abs_diff={diff.max().item():.6f} mean_abs_diff={diff.mean().item():.6f}")


if __name__ == "__main__":
    main()
