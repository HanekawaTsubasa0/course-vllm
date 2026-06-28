from __future__ import annotations

import argparse
import asyncio
import json
import time
from statistics import mean

import httpx


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18080/generate")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json", action="store_true", help="print machine-readable metrics")
    args = parser.parse_args()

    results: list[dict] = []
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=None) as client:
        tasks = [
            asyncio.create_task(send_one(client, semaphore, args, results))
            for _ in range(args.num_requests)
        ]
        await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started
    metrics = summarize(results, elapsed, args)
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}={value:.6f}")
            else:
                print(f"{key}={value}")


async def send_one(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, args, results: list[dict]) -> None:
    payload = {
        "prompt": args.prompt,
        "stream": False,
        "sampling_params": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
    }
    async with semaphore:
        started = time.perf_counter()
        response = await client.post(args.url, json=payload)
        response.raise_for_status()
        latency = time.perf_counter() - started
        data = response.json()
        token_ids = data.get("token_ids", [])
        results.append(
            {
                "latency_s": latency,
                "num_output_tokens": len(token_ids),
                "finish_reason": data.get("finish_reason", "unknown"),
            }
        )


def summarize(results: list[dict], elapsed_s: float, args) -> dict:
    latencies = sorted(result["latency_s"] for result in results)
    output_tokens = sum(result["num_output_tokens"] for result in results)
    completed = len(results)
    return {
        "requests": args.num_requests,
        "completed": completed,
        "concurrency": args.concurrency,
        "elapsed_s": elapsed_s,
        "requests_per_s": completed / elapsed_s if elapsed_s > 0 else 0.0,
        "output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "latency_avg_s": mean(latencies) if latencies else 0.0,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p90_s": percentile(latencies, 0.90),
        "latency_p99_s": percentile(latencies, 0.99),
        "latency_max_s": max(latencies) if latencies else 0.0,
        "estimated_tpot_s": sum(latencies) / output_tokens if output_tokens > 0 else 0.0,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[index]


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
