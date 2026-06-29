from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import sys
import time
from typing import Any

import httpx


DEFAULT_URL = "http://127.0.0.1:18080/v1/chat/completions"
DEFAULT_PROMPTS = [
    "用两句话解释 prefill 和 decode 的区别。",
    "用两句话解释 paged KV cache 的作用。",
    "用两句话解释 continuous batching 为什么能提升吞吐。",
]


def main() -> None:
    args = parse_args()
    prompts = args.prompt or DEFAULT_PROMPTS
    events: mp.Queue[tuple[int, str, str]] = mp.Queue()
    processes: list[mp.Process] = []

    for client_id, prompt in enumerate(prompts, 1):
        process = mp.Process(
            target=run_client,
            args=(client_id, args.url, prompt, args.max_tokens, args.temperature, args.top_k, args.timeout, events),
        )
        process.start()
        processes.append(process)

    alive = len(processes)
    while alive:
        try:
            client_id, event_type, text = events.get(timeout=0.1)
        except queue.Empty:
            alive = sum(process.is_alive() for process in processes)
            continue
        if event_type == "start":
            print(f"\n[client-{client_id}] prompt: {text}", flush=True)
        elif event_type == "token":
            print(f"[client-{client_id}] {text}", end="", flush=True)
        elif event_type == "done":
            print(f"\n[client-{client_id}] done: {text}", flush=True)
        elif event_type == "error":
            print(f"\n[client-{client_id}] error: {text}", file=sys.stderr, flush=True)

    for process in processes:
        process.join()

    failed = [index + 1 for index, process in enumerate(processes) if process.exitcode != 0]
    if failed:
        raise SystemExit(f"failed clients: {failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run several streaming chat clients against one course-vllm server.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--prompt", action="append", help="prompt for one client; repeat to override defaults")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    return parser.parse_args()


def run_client(
    client_id: int,
    url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    timeout: float | None,
    events: mp.Queue[tuple[int, str, str]],
) -> None:
    events.put((client_id, "start", prompt))
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "sampling_params": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
        },
    }
    text_parts: list[str] = []
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for event in iter_sse(response.iter_lines()):
                    if event.get("event") == "token":
                        token = str(event.get("text", ""))
                        text_parts.append(token)
                        events.put((client_id, "token", token))
    except Exception as exc:
        events.put((client_id, "error", repr(exc)))
        raise
    events.put((client_id, "done", f"{len(''.join(text_parts))} chars"))


def iter_sse(lines: Any):
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.strip()
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            return
        yield json.loads(data)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    start = time.time()
    try:
        main()
    finally:
        elapsed = time.time() - start
        print(f"\nelapsed: {elapsed:.2f}s", flush=True)
