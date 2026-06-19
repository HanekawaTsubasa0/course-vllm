from __future__ import annotations

import argparse
import json

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    messages: list[dict[str, str]] = []
    with httpx.Client(timeout=None) as client:
        while True:
            try:
                user = input("user> ").strip()
            except EOFError:
                print()
                return
            if not user:
                continue
            if user in {"/exit", "/quit"}:
                return
            messages.append({"role": "user", "content": user})
            payload = {
                "messages": messages,
                "stream": True,
                "sampling_params": {"temperature": 0.6, "max_tokens": args.max_tokens},
            }
            assistant_chunks: list[str] = []
            print("assistant> ", end="", flush=True)
            with client.stream("POST", args.url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ")
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("event") == "token":
                        text = event["text"]
                        assistant_chunks.append(text)
                        print(text, end="", flush=True)
            print()
            messages.append({"role": "assistant", "content": "".join(assistant_chunks)})


if __name__ == "__main__":
    main()
