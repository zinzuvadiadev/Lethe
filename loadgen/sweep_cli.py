from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from configs.loader import load_hardware_config, load_model_config
from loadgen.aggressiveness import DEFAULT_SETTINGS
from loadgen.sweep_orchestrator import run_aggressiveness_sweep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full aggressiveness sweep: launch a server per "
        "setting, load-test it, tear it down, repeat."
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--slo-latency-sec", type=float, default=5.0)
    parser.add_argument("--out-dir", default="results/raw")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-ready-timeout-sec", type=float, default=400.0)
    args = parser.parse_args()

    model_cfg = load_model_config(args.model_config)
    hw_cfg = load_hardware_config(args.hardware_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = asyncio.run(
        run_aggressiveness_sweep(
            model_cfg, hw_cfg, DEFAULT_SETTINGS,
            rate_per_sec=args.rate, duration_sec=args.duration,
            slo_latency_sec=args.slo_latency_sec, output_dir=out_dir,
            port=args.port, server_ready_timeout_sec=args.server_ready_timeout_sec,
        )
    )
    print(f"wrote {len(paths)} sweep files:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
