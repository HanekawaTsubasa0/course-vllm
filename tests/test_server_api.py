from course_vllm.server import api


class FakeEngine:
    def __init__(
        self,
        model,
        *,
        dtype,
        device,
        backend,
        stage,
        kernel_impl,
        use_pinned_memory=False,
        use_transfer_stream=False,
    ):
        self.model = model
        self.backend = type("Backend", (), {"apply_chat_template": lambda self, messages: "chat"})()
        self._info = {
            "backend": backend,
            "stage": {"key": stage},
            "kernel_impl": kernel_impl,
            "model_backend": "FakeBackend",
            "system_optimizations": {
                "pinned_memory": use_pinned_memory,
                "transfer_stream": use_transfer_stream,
            },
        }

    def info(self):
        return self._info


def test_create_app_health_reports_stage_and_kernel_impl(monkeypatch):
    monkeypatch.setattr(api, "Engine", FakeEngine)
    app = api.create_app(
        "fake-model",
        backend="course",
        stage="week04",
        kernel_impl="auto",
        use_pinned_memory=True,
        use_transfer_stream=True,
        max_batched_tokens=64,
        max_queue_size=4,
        max_prompt_chars=128,
        enable_chunked_prefill=True,
        cache_aware_scheduling=True,
    )

    health = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/health")()

    assert health["status"] == "ok"
    assert health["engine"]["stage"]["key"] == "week04"
    assert health["engine"]["kernel_impl"] == "auto"
    assert health["engine"]["system_optimizations"]["pinned_memory"] is True
    assert health["max_batched_tokens"] == 64
    assert health["batching"]["max_queue_size"] == 4
    assert health["batching"]["enable_chunked_prefill"] is True
    assert health["batching"]["cache_aware_scheduling"] is True
