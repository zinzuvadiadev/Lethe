from __future__ import annotations

import dataclasses
import time

import httpx


@dataclasses.dataclass(frozen=True)
class RequestResult:
    context_tokens: int
    requested_output_tokens: int
    completed_output_tokens: int
    latency_sec: float
    ttft_sec: float | None
    success: bool
    error: str | None


async def send_completion_request(
    client: httpx.AsyncClient,
    base_url: str,
    served_model_name: str,
    prompt: str,
    context_tokens: int,
    max_tokens: int,
    timeout_sec: float,
) -> RequestResult:
    start = time.perf_counter()
    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/completions",
            json={
                "model": served_model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": True,
            },
            timeout=timeout_sec,
        ) as response:
            response.raise_for_status()
            ttft: float | None = None
            completed_tokens = 0
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                if ttft is None:
                    ttft = time.perf_counter() - start
                completed_tokens += 1
        return RequestResult(
            context_tokens=context_tokens,
            requested_output_tokens=max_tokens,
            completed_output_tokens=completed_tokens,
            latency_sec=time.perf_counter() - start,
            ttft_sec=ttft,
            success=True,
            error=None,
        )
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        return RequestResult(
            context_tokens=context_tokens,
            requested_output_tokens=max_tokens,
            completed_output_tokens=0,
            latency_sec=time.perf_counter() - start,
            ttft_sec=None,
            success=False,
            error=str(exc),
        )
