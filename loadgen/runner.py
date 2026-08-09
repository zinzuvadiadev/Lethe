from __future__ import annotations

import asyncio
import csv
import dataclasses
from pathlib import Path

import httpx

from loadgen.arrivals import poisson_arrival_times
from loadgen.client import RequestResult, send_completion_request
from loadgen.sampling import DEFAULT_BUCKETS, sample_context_length, sample_output_length


@dataclasses.dataclass(frozen=True)
class SweepConfig:
    base_url: str
    served_model_name: str
    rate_per_sec: float
    duration_sec: float
    eviction_aggressiveness: str
    slo_latency_sec: float
    timeout_sec: float = 30.0


CSV_FIELDS = [
    "eviction_aggressiveness",
    "rate_per_sec",
    "context_tokens",
    "requested_output_tokens",
    "completed_output_tokens",
    "latency_sec",
    "ttft_sec",
    "success",
    "met_slo",
    "error",
]


def result_to_row(result: RequestResult, cfg: SweepConfig) -> dict:
    return {
        "eviction_aggressiveness": cfg.eviction_aggressiveness,
        "rate_per_sec": cfg.rate_per_sec,
        "context_tokens": result.context_tokens,
        "requested_output_tokens": result.requested_output_tokens,
        "completed_output_tokens": result.completed_output_tokens,
        "latency_sec": result.latency_sec,
        "ttft_sec": result.ttft_sec,
        "success": result.success,
        "met_slo": result.success and result.latency_sec <= cfg.slo_latency_sec,
        "error": result.error,
    }


async def run_sweep(cfg: SweepConfig) -> list[dict]:
    arrival_times = poisson_arrival_times(cfg.rate_per_sec, cfg.duration_sec)

    async with httpx.AsyncClient() as client:

        async def issue_at(delay: float) -> dict:
            await asyncio.sleep(delay)
            context_tokens = sample_context_length(DEFAULT_BUCKETS)
            output_tokens = sample_output_length()
            prompt = "lorem " * context_tokens
            result = await send_completion_request(
                client,
                cfg.base_url,
                cfg.served_model_name,
                prompt,
                context_tokens,
                output_tokens,
                cfg.timeout_sec,
            )
            return result_to_row(result, cfg)

        rows = await asyncio.gather(*(issue_at(t) for t in arrival_times))
    return list(rows)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
