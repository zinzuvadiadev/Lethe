from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from configs.loader import load_model_config
from loadgen.runner import SweepConfig, run_sweep, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a load-test sweep against a vLLM server")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--eviction-aggressiveness", default="baseline")
    parser.add_argument("--slo-latency-sec", type=float, default=5.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model_cfg = load_model_config(args.model_config)
    cfg = SweepConfig(
        base_url=args.base_url,
        served_model_name=model_cfg.served_model_name,
        rate_per_sec=args.rate,
        duration_sec=args.duration,
        eviction_aggressiveness=args.eviction_aggressiveness,
        slo_latency_sec=args.slo_latency_sec,
    )
    rows = asyncio.run(run_sweep(cfg))
    write_csv(rows, Path(args.out))
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
