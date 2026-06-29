from __future__ import annotations

import argparse
import asyncio
import json
import time
from statistics import mean
from typing import Any

import httpx


DEFAULT_URL = "http://127.0.0.1:18080/generate"


async def main_async() -> None:
    args = parse_args()
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        health_before = await get_health(client, args.url)
        print_health("before", health_before)

        sequential = await run_case(
            client,
            url=args.url,
            name="sequential_no_batch",
            num_requests=args.num_requests,
            concurrency=1,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        health_after_sequential = await get_health(client, args.url)
        print_metrics(sequential)
        print_batch_delta("sequential_no_batch", health_before, health_after_sequential)

        concurrent = await run_case(
            client,
            url=args.url,
            name="concurrent_batching",
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        health_after_concurrent = await get_health(client, args.url)
        print_metrics(concurrent)
        print_batch_delta("concurrent_batching", health_after_sequential, health_after_concurrent)

    speedup = sequential["elapsed_s"] / concurrent["elapsed_s"] if concurrent["elapsed_s"] > 0 else 0.0
    throughput_ratio = (
        concurrent["requests_per_s"] / sequential["requests_per_s"]
        if sequential["requests_per_s"] > 0
        else 0.0
    )
    print("\n==============================")
    print("COMPARISON")
    print("==============================")
    print(f"elapsed_speedup={speedup:.3f}x")
    print(f"throughput_ratio={throughput_ratio:.3f}x")
    if speedup > 1:
        print("result=concurrent batching finished faster")
    else:
        print("result=concurrent batching was not faster on this workload")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sequential non-streaming requests with batched concurrent requests.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--num-requests", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--prompt", default="用两句话解释 paged KV cache。")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=None)
    return parser.parse_args()


async def run_case(
    client: httpx.AsyncClient,
    *,
    url: str,
    name: str,
    num_requests: int,
    concurrency: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    print("\n==============================")
    print(f"RUN {name}")
    print("==============================")
    results: list[dict[str, Any]] = []
    semaphore = asyncio.Semaphore(concurrency)
    started = time.perf_counter()
    tasks = [
        asyncio.create_task(
            send_one(
                client,
                semaphore,
                url=url,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                results=results,
                request_id=index + 1,
            )
        )
        for index in range(num_requests)
    ]
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started
    return summarize(name, results, elapsed, num_requests, concurrency)


async def send_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    results: list[dict[str, Any]],
    request_id: int,
) -> None:
    payload = {
        "prompt": f"{prompt}\n请求编号：{request_id}",
        "stream": False,
        "sampling_params": {
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }
    async with semaphore:
        started = time.perf_counter()
        response = await client.post(url, json=payload)
        response.raise_for_status()
        elapsed = time.perf_counter() - started
    data = response.json()
    token_ids = data.get("token_ids", [])
    print(f"request-{request_id}: latency={elapsed:.3f}s tokens={len(token_ids)} finish={data.get('finish_reason')}")
    results.append(
        {
            "latency_s": elapsed,
            "num_output_tokens": len(token_ids),
            "finish_reason": data.get("finish_reason", "unknown"),
        }
    )


def summarize(
    name: str,
    results: list[dict[str, Any]],
    elapsed_s: float,
    num_requests: int,
    concurrency: int,
) -> dict[str, Any]:
    latencies = sorted(result["latency_s"] for result in results)
    output_tokens = sum(result["num_output_tokens"] for result in results)
    completed = len(results)
    return {
        "name": name,
        "requests": num_requests,
        "completed": completed,
        "concurrency": concurrency,
        "elapsed_s": elapsed_s,
        "requests_per_s": completed / elapsed_s if elapsed_s > 0 else 0.0,
        "output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "latency_avg_s": mean(latencies) if latencies else 0.0,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p90_s": percentile(latencies, 0.90),
        "latency_max_s": max(latencies) if latencies else 0.0,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[index]


async def get_health(client: httpx.AsyncClient, generate_url: str) -> dict[str, Any]:
    base = generate_url.removesuffix("/generate").rstrip("/")
    response = await client.get(f"{base}/health")
    response.raise_for_status()
    return response.json()


def print_health(label: str, health: dict[str, Any]) -> None:
    batching = health.get("batching", {})
    print("\n==============================")
    print(f"HEALTH {label}")
    print("==============================")
    print(json.dumps({"batching": batching}, ensure_ascii=False, indent=2))


def print_metrics(metrics: dict[str, Any]) -> None:
    print("\nMETRICS")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}={value:.6f}")
        else:
            print(f"{key}={value}")


def print_batch_delta(label: str, before: dict[str, Any], after: dict[str, Any]) -> None:
    before_batching = before.get("batching", {})
    after_batching = after.get("batching", {})
    total_requests_delta = after_batching.get("total_requests", 0) - before_batching.get("total_requests", 0)
    total_batches_delta = after_batching.get("total_batches", 0) - before_batching.get("total_batches", 0)
    average_batch_size = total_requests_delta / total_batches_delta if total_batches_delta > 0 else 0.0
    print("\nBATCHING DELTA")
    print(f"name={label}")
    print(f"total_requests_delta={total_requests_delta}")
    print(f"total_batches_delta={total_batches_delta}")
    print(f"average_batch_size_delta={average_batch_size:.3f}")
    print(f"max_observed_batch_size={after_batching.get('max_observed_batch_size')}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
