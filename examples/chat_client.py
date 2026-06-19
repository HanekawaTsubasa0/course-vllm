from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shlex
import sys
from collections.abc import Iterable

import httpx


DEFAULT_URL = "http://127.0.0.1:18080/v1/chat/completions"


@dataclass
class ChatConfig:
    max_tokens: int | None = None
    temperature: float = 0.6
    top_k: int | None = None
    stream: bool = True


class ChatSession:
    def __init__(
        self,
        client: httpx.Client,
        *,
        url: str,
        config: ChatConfig,
        system_prompt: str | None = None,
    ):
        self.client = client
        self.url = url
        self.config = config
        self.messages: list[dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def run_once(self, prompt: str) -> None:
        self.send(prompt)

    def run_repl(self) -> None:
        print("course-vllm chat client")
        print("commands: /help, /health, /params, /clear, /save [path], /load <path>, /exit")
        while True:
            try:
                text = input("user> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not text:
                continue
            if text.startswith("/"):
                if self.handle_command(text) == "exit":
                    return
                continue
            self.send(text)

    def send(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        try:
            answer = self._request()
        except httpx.HTTPError as exc:
            self.messages.pop()
            print(f"request failed: {exc}", file=sys.stderr)
            return
        except KeyboardInterrupt:
            self.messages.pop()
            print("\ninterrupted", file=sys.stderr)
            return
        self.messages.append({"role": "assistant", "content": answer})

    def handle_command(self, line: str) -> str | None:
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            print(f"bad command: {exc}")
            return None
        command = parts[0]
        args = parts[1:]
        if command in {"/exit", "/quit"}:
            return "exit"
        if command == "/help":
            self._print_help()
        elif command == "/clear":
            self._clear()
        elif command == "/health":
            self._print_health()
        elif command == "/history":
            self._print_history(int(args[0]) if args else 10)
        elif command == "/save":
            self._save(Path(args[0]) if args else Path("chat_history.json"))
        elif command == "/load":
            if not args:
                print("usage: /load <path>")
            else:
                self._load(Path(args[0]))
        elif command == "/system":
            self._set_system(" ".join(args))
        elif command == "/params":
            print(json.dumps(asdict(self.config), indent=2, ensure_ascii=False))
        elif command == "/set":
            self._set_param(args)
        elif command == "/stream":
            self._set_stream(args)
        else:
            print(f"unknown command: {command}")
            print("type /help for available commands")
        return None

    def _request(self) -> str:
        payload = {
            "messages": self.messages,
            "stream": self.config.stream,
            "sampling_params": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_k": self.config.top_k,
            },
        }
        print("assistant> ", end="", flush=True)
        if self.config.stream:
            answer = self._stream(payload)
        else:
            response = self.client.post(self.url, json=payload)
            response.raise_for_status()
            answer = response.json()["text"]
            print(answer)
        return answer

    def _stream(self, payload: dict) -> str:
        chunks: list[str] = []
        with self.client.stream("POST", self.url, json=payload) as response:
            response.raise_for_status()
            for event in iter_sse(response.iter_lines()):
                if event.get("event") == "token":
                    token = event["text"]
                    chunks.append(token)
                    print(token, end="", flush=True)
        print()
        return "".join(chunks)

    def _print_health(self) -> None:
        response = self.client.get(self._health_url())
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))

    def _health_url(self) -> str:
        base = self.url.removesuffix("/v1/chat/completions").rstrip("/")
        return f"{base}/health"

    def _clear(self) -> None:
        system = next((msg for msg in self.messages if msg["role"] == "system"), None)
        self.messages[:] = [system] if system else []
        print("history cleared")

    def _print_history(self, limit: int) -> None:
        recent = self.messages[-max(limit, 1) :]
        for index, message in enumerate(recent, 1):
            text = message["content"].replace("\n", " ")
            print(f"{index}. {message['role']}: {text[:120]}")

    def _save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"config": asdict(self.config), "messages": self.messages}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"saved {path}")

    def _load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        messages = data["messages"] if isinstance(data, dict) else data
        self.messages = _validate_messages(messages)
        if isinstance(data, dict) and "config" in data:
            self.config = ChatConfig(**{**asdict(self.config), **data["config"]})
        print(f"loaded {len(self.messages)} messages")

    def _set_system(self, content: str) -> None:
        self.messages = [msg for msg in self.messages if msg["role"] != "system"]
        if content:
            self.messages.insert(0, {"role": "system", "content": content})
            print("system prompt updated")
        else:
            print("system prompt cleared")

    def _set_param(self, args: list[str]) -> None:
        if len(args) != 2:
            print("usage: /set max_tokens|temperature|top_k <value>")
            return
        name, value = args
        if name == "max_tokens":
            self.config.max_tokens = None if value.lower() in {"none", "null", "off"} else max(1, int(value))
        elif name == "temperature":
            self.config.temperature = max(0.0, float(value))
        elif name == "top_k":
            self.config.top_k = None if value.lower() in {"none", "null", "off"} else max(1, int(value))
        else:
            print(f"unknown parameter: {name}")
            return
        print(json.dumps(asdict(self.config), indent=2, ensure_ascii=False))

    def _set_stream(self, args: list[str]) -> None:
        if len(args) != 1 or args[0].lower() not in {"on", "off"}:
            print("usage: /stream on|off")
            return
        self.config.stream = args[0].lower() == "on"
        print(f"stream={self.config.stream}")

    def _print_help(self) -> None:
        print(
            "\n".join(
                [
                    "/help                 show commands",
                    "/health               show server health and batching counters",
                    "/params               show sampling parameters",
                    "/set max_tokens 64    set max generated tokens; use none/off for no limit",
                    "/set temperature 0    change temperature",
                    "/set top_k 40         change top-k; use none/off to disable",
                    "/stream on|off        enable or disable SSE streaming",
                    "/system <text>        replace system prompt; empty clears it",
                    "/history [n]          show recent messages",
                    "/clear                clear chat history except system prompt",
                    "/save [path]          save config and chat history",
                    "/load <path>          load config and chat history",
                    "/exit                 quit",
                ]
            )
        )


def iter_sse(lines: Iterable[str]):
    for line in lines:
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            break
        yield json.loads(data)


def _validate_messages(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("history must be a list of messages")
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"bad message: {item!r}")
        content = item.get("content")
        if not isinstance(content, str):
            raise ValueError(f"bad message content: {item!r}")
        messages.append({"role": item["role"], "content": content})
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--max-tokens", type=int, default=None, help="maximum generated tokens; omit for no explicit limit")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--system", default=None)
    parser.add_argument("--load", type=Path, default=None)
    parser.add_argument("--once", default=None, help="send one prompt and exit")
    parser.add_argument("--timeout", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ChatConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        stream=not args.no_stream,
    )
    with httpx.Client(timeout=args.timeout) as client:
        session = ChatSession(client, url=args.url, config=config, system_prompt=args.system)
        if args.load:
            session._load(args.load)
        if args.once is not None:
            session.run_once(args.once)
        else:
            session.run_repl()


if __name__ == "__main__":
    main()
