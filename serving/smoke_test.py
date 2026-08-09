from __future__ import annotations

import argparse

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one completion request to a running vLLM server")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--served-model-name", default="qwen3-4b-instruct-2507")
    args = parser.parse_args()

    response = httpx.post(
        f"{args.base_url}/v1/completions",
        json={
            "model": args.served_model_name,
            "prompt": "The capital of France is",
            "max_tokens": 16,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    body = response.json()
    text = body["choices"][0]["text"]
    print(f"status: {response.status_code}")
    print(f"completion: {text!r}")
    assert text.strip(), "expected non-empty completion text"
    print("OK: server responded with a non-empty completion")


if __name__ == "__main__":
    main()
