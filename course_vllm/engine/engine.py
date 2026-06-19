from __future__ import annotations

from collections.abc import Iterator

from course_vllm.engine.request import Request, Sequence
from course_vllm.engine.sampler import Sampler, SamplingParams
from course_vllm.model.hf_backend import HFModelBackend
from course_vllm.model.qwen3_backend import Qwen3TorchBackend


class Engine:
    def __init__(
        self,
        model: str,
        *,
        dtype: str = "bfloat16",
        device: str | None = None,
        backend: str = "hf",
    ):
        if backend == "hf":
            self.backend = HFModelBackend(model, dtype=dtype, device=device)
        elif backend == "course":
            self.backend = Qwen3TorchBackend(model, dtype=dtype, device=device)
        else:
            raise ValueError(f"unsupported backend: {backend}")
        self.backend_name = backend

    @property
    def tokenizer(self):
        return self.backend.tokenizer

    def generate(self, prompt: str, sampling_params: SamplingParams | None = None) -> dict:
        sampling_params = sampling_params or SamplingParams()
        text = ""
        token_ids: list[int] = []
        for event in self.generate_stream(prompt, sampling_params):
            if event["event"] == "token":
                text += event["text"]
                token_ids.append(event["token_id"])
            elif event["event"] == "finished":
                return {
                    "text": text,
                    "token_ids": token_ids,
                    "finish_reason": event["finish_reason"],
                }
        return {"text": text, "token_ids": token_ids, "finish_reason": "unknown"}

    def generate_stream(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ) -> Iterator[dict]:
        sampling_params = sampling_params or SamplingParams()
        request = Request(prompt=prompt, sampling_params=sampling_params)
        prompt_token_ids = self.backend.encode(prompt)
        if not prompt_token_ids:
            raise ValueError("prompt encoded to an empty token sequence")

        seq = Sequence(request=request, prompt_token_ids=prompt_token_ids)
        sampler = Sampler(sampling_params)

        output = self.backend.prefill(seq.prompt_token_ids)
        past_key_values = output.past_key_values
        next_token_id = sampler.sample(output.logits)

        try:
            while True:
                seq.append_token(next_token_id)
                token_text = self.backend.decode([next_token_id])
                yield {
                    "event": "token",
                    "request_id": seq.request_id,
                    "token_id": next_token_id,
                    "text": token_text,
                }

                finish_reason = self._finish_reason(seq, next_token_id, sampling_params)
                if finish_reason is not None:
                    seq.finish()
                    yield {
                        "event": "finished",
                        "request_id": seq.request_id,
                        "finish_reason": finish_reason,
                        "token_ids": seq.generated_token_ids,
                        "text": self.backend.decode(seq.generated_token_ids),
                    }
                    return

                output = self.backend.decode_step(next_token_id, past_key_values)
                past_key_values = output.past_key_values
                next_token_id = sampler.sample(output.logits)
        finally:
            self._release_cache(past_key_values)

    def chat(self, messages: list[dict[str, str]], sampling_params: SamplingParams | None = None) -> dict:
        prompt = self.backend.apply_chat_template(messages)
        return self.generate(prompt, sampling_params)

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        sampling_params: SamplingParams | None = None,
    ) -> Iterator[dict]:
        prompt = self.backend.apply_chat_template(messages)
        yield from self.generate_stream(prompt, sampling_params)

    def _finish_reason(
        self,
        seq: Sequence,
        token_id: int,
        sampling_params: SamplingParams,
    ) -> str | None:
        if token_id == self.backend.eos_token_id:
            return "eos"
        if token_id in sampling_params.stop_token_ids:
            return "stop"
        if seq.reached_max_tokens():
            return "length"
        return None

    def _release_cache(self, past_key_values: object | None) -> None:
        release_cache = getattr(self.backend, "release_cache", None)
        if release_cache is not None:
            release_cache(past_key_values)
