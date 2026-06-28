from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from course_vllm.engine.request import RequestStatus, Sequence


class BatchKind(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(slots=True)
class ScheduledBatch:
    kind: BatchKind
    sequences: list[Sequence]
    num_tokens: int


class Scheduler:
    """Single-process scheduler skeleton for prefill/decode batching."""

    def __init__(
        self,
        *,
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 2048,
        enable_chunked_prefill: bool = False,
    ):
        if max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be > 0")
        if max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be > 0")
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        seq.status = RequestStatus.WAITING
        seq.request.status = RequestStatus.WAITING
        self.waiting.append(seq)

    def schedule(self) -> ScheduledBatch | None:
        prefill = self._schedule_prefill()
        if prefill is not None:
            return prefill
        return self._schedule_decode()

    def _schedule_prefill(self) -> ScheduledBatch | None:
        selected: list[Sequence] = []
        token_budget = 0
        while self.waiting and len(selected) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = len(seq.prompt_token_ids) - seq.prefill_offset
            if remaining <= 0:
                self.waiting.popleft()
                seq.status = RequestStatus.RUNNING
                seq.request.status = RequestStatus.RUNNING
                self.running.append(seq)
                continue
            available = self.max_num_batched_tokens - token_budget
            if available <= 0:
                break
            if remaining > available and (selected or not self.enable_chunked_prefill):
                break
            scheduled_tokens = min(remaining, available) if self.enable_chunked_prefill else remaining
            seq.scheduled_start = seq.prefill_offset
            seq.scheduled_end = seq.prefill_offset + scheduled_tokens
            self.waiting.popleft()
            if seq.scheduled_end < len(seq.prompt_token_ids):
                self.waiting.appendleft(seq)
            else:
                seq.status = RequestStatus.RUNNING
                seq.request.status = RequestStatus.RUNNING
                self.running.append(seq)
            selected.append(seq)
            token_budget += scheduled_tokens
        if not selected:
            return None
        return ScheduledBatch(kind=BatchKind.PREFILL, sequences=selected, num_tokens=token_budget)

    def _schedule_decode(self) -> ScheduledBatch | None:
        active = [seq for seq in self.running if seq.status == RequestStatus.RUNNING]
        if not active:
            return None
        selected = active[: self.max_num_seqs]
        return ScheduledBatch(kind=BatchKind.DECODE, sequences=selected, num_tokens=len(selected))

    def finish(self, seq: Sequence) -> None:
        seq.finish()
        self.running = [item for item in self.running if item.request_id != seq.request_id]

    def preempt(self, seq: Sequence) -> None:
        self.running = [item for item in self.running if item.request_id != seq.request_id]
        seq.status = RequestStatus.WAITING
        seq.request.status = RequestStatus.WAITING
        seq.past_key_values = None
        seq.next_token_id = None
        seq.prefill_offset = 0
        seq.scheduled_start = 0
        seq.scheduled_end = 0
        self.waiting.appendleft(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)
