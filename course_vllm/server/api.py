from __future__ import annotations

import argparse
import json
from collections.abc import Iterator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from course_vllm.engine.engine import Engine
from course_vllm.engine.sampler import SamplingParams
from course_vllm.server.protocol import (
    ChatCompletionRequest,
    GenerateRequest,
    GenerateResponse,
)


def create_app(model: str, *, dtype: str = "bfloat16", device: str | None = None) -> FastAPI:
    app = FastAPI(title="course-vllm")
    engine = Engine(model=model, dtype=dtype, device=device)
    app.state.engine = engine

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model}

    @app.post("/generate")
    def generate(request: GenerateRequest):
        params = _sampling_params(request.sampling_params)
        if request.stream:
            return StreamingResponse(
                _sse(engine.generate_stream(request.prompt, params)),
                media_type="text/event-stream",
            )
        result = engine.generate(request.prompt, params)
        return GenerateResponse(**result)

    @app.post("/v1/chat/completions")
    def chat(request: ChatCompletionRequest):
        params = _sampling_params(request.sampling_params)
        messages = [message.model_dump() for message in request.messages]
        if request.stream:
            return StreamingResponse(
                _sse(engine.chat_stream(messages, params)),
                media_type="text/event-stream",
            )
        result = engine.chat(messages, params)
        return GenerateResponse(**result)

    return app


def _sampling_params(params) -> SamplingParams:
    return SamplingParams(
        temperature=params.temperature,
        max_tokens=params.max_tokens,
        top_k=params.top_k,
        seed=params.seed,
    )


def _sse(events: Iterator[dict]) -> Iterator[str]:
    for event in events:
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    app = create_app(args.model, dtype=args.dtype, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
