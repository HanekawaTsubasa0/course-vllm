from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

from course_vllm.engine.engine import Engine
from course_vllm.engine.sampler import SamplingParams


@dataclass(frozen=True, slots=True)
class _QueueItem:
    prompt: str
    sampling_params: SamplingParams
    future: asyncio.Future


class BatchingEngine:
    """Async request queue that feeds non-streaming HTTP requests into Engine.generate_batch."""

    def __init__(
        self,
        engine: Engine,
        *,
        max_batch_size: int = 8,
        batch_wait_ms: float = 2.0,
        max_num_batched_tokens: int = 2048,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be >= 0")
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.max_num_batched_tokens = max_num_batched_tokens
        self._queue: asyncio.Queue[_QueueItem | None] = asyncio.Queue()
        self._pending: deque[_QueueItem | None] = deque()
        self._worker: asyncio.Task | None = None

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker is None:
            return
        await self._queue.put(None)
        await self._worker
        self._worker = None

    async def generate(self, prompt: str, sampling_params: SamplingParams) -> dict:
        await self.start()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put(_QueueItem(prompt=prompt, sampling_params=sampling_params, future=future))
        return await future

    async def _run(self) -> None:
        while True:
            item = await self._next_item()
            if item is None:
                return
            batch = [item]
            await self._collect_batch(batch)
            await self._process_batch(batch)

    async def _collect_batch(self, batch: list[_QueueItem]) -> None:
        if self.batch_wait_s > 0:
            await asyncio.sleep(self.batch_wait_s)
        while len(batch) < self.max_batch_size:
            try:
                item = self._next_item_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                self._pending.appendleft(None)
                break
            if item.sampling_params != batch[0].sampling_params:
                self._pending.appendleft(item)
                break
            batch.append(item)

    async def _process_batch(self, batch: list[_QueueItem]) -> None:
        try:
            results = self.engine.generate_batch(
                [item.prompt for item in batch],
                batch[0].sampling_params,
                max_num_seqs=self.max_batch_size,
                max_num_batched_tokens=self.max_num_batched_tokens,
            )
        except Exception as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
            return
        for item, result in zip(batch, results):
            if not item.future.done():
                item.future.set_result(result)

    async def _next_item(self) -> _QueueItem | None:
        if self._pending:
            return self._pending.popleft()
        return await self._queue.get()

    def _next_item_nowait(self) -> _QueueItem | None:
        if self._pending:
            return self._pending.popleft()
        return self._queue.get_nowait()
