from __future__ import annotations

import argparse
import asyncio
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
    args = parser.parse_args()

    latencies: list[float] = []
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=None) as client:
        tasks = [
            asyncio.create_task(send_one(client, semaphore, args, latencies))
            for _ in range(args.num_requests)
        ]
        await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started
    print(f"requests={args.num_requests}")
    print(f"concurrency={args.concurrency}")
    print(f"elapsed_s={elapsed:.3f}")
    print(f"requests_per_s={args.num_requests / elapsed:.3f}")
    print(f"latency_avg_s={mean(latencies):.3f}")
    print(f"latency_max_s={max(latencies):.3f}")


async def send_one(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, args, latencies: list[float]) -> None:
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
        latencies.append(time.perf_counter() - started)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
