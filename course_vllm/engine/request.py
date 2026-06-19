from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from time import perf_counter

from course_vllm.engine.sampler import SamplingParams


_REQUEST_IDS = count()


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(slots=True)
class Request:
    prompt: str
    sampling_params: SamplingParams
    request_id: int = field(default_factory=lambda: next(_REQUEST_IDS))
    created_at: float = field(default_factory=perf_counter)
    status: RequestStatus = RequestStatus.WAITING


@dataclass(slots=True)
class Sequence:
    request: Request
    prompt_token_ids: list[int]
    generated_token_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    past_key_values: object | None = None
    next_token_id: int | None = None
    finish_reason: str | None = None

    @property
    def request_id(self) -> int:
        return self.request.request_id

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.generated_token_ids

    def append_token(self, token_id: int) -> None:
        self.generated_token_ids.append(int(token_id))

    def reached_max_tokens(self) -> bool:
        max_tokens = self.request.sampling_params.max_tokens
        return max_tokens is not None and len(self.generated_token_ids) >= max_tokens

    def finish(self) -> None:
        self.status = RequestStatus.FINISHED
        self.request.status = RequestStatus.FINISHED
