import json

import httpx

from examples.chat_client import ChatConfig, ChatSession, iter_sse


def test_iter_sse_parses_token_events():
    events = list(
        iter_sse(
            [
                ": ping",
                'data: {"event":"token","text":"hi"}',
                "data: [DONE]",
            ]
        )
    )
    assert events == [{"event": "token", "text": "hi"}]


def test_chat_session_save_and_load(tmp_path):
    session = ChatSession(httpx.Client(), url="http://test/v1/chat/completions", config=ChatConfig(), system_prompt="sys")
    session.messages.append({"role": "user", "content": "hello"})
    path = tmp_path / "history.json"
    session._save(path)

    loaded = ChatSession(httpx.Client(), url="http://test/v1/chat/completions", config=ChatConfig())
    loaded._load(path)

    assert loaded.messages == session.messages
    assert json.loads(path.read_text())["config"]["stream"] is True


def test_chat_session_set_params():
    session = ChatSession(httpx.Client(), url="http://test/v1/chat/completions", config=ChatConfig())
    session.handle_command("/set max_tokens 4")
    assert session.config.max_tokens == 4
    session.handle_command("/set max_tokens none")
    session.handle_command("/set temperature 0")
    session.handle_command("/set top_k none")
    session.handle_command("/stream off")

    assert session.config.max_tokens is None
    assert session.config.temperature == 0
    assert session.config.top_k is None
    assert session.config.stream is False


def test_chat_session_non_stream_request_uses_chat_endpoint():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["sampling_params"]["max_tokens"] is None
        assert body["messages"][-1] == {"role": "user", "content": "hello"}
        return httpx.Response(200, json={"text": "world", "token_ids": [1], "finish_reason": "length"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    session = ChatSession(client, url="http://test/v1/chat/completions", config=ChatConfig(stream=False))
    session.send("hello")

    assert requests[0].url.path == "/v1/chat/completions"
    assert session.messages[-1] == {"role": "assistant", "content": "world"}


def test_chat_session_stream_request_collects_tokens():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        return httpx.Response(
            200,
            text='data: {"event":"token","text":"wo"}\n\ndata: {"event":"token","text":"rld"}\n\ndata: [DONE]\n\n',
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    session = ChatSession(client, url="http://test/v1/chat/completions", config=ChatConfig(stream=True))
    session.send("hello")

    assert session.messages[-1] == {"role": "assistant", "content": "world"}
