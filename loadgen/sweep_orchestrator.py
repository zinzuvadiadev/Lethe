from __future__ import annotations

from pathlib import Path

from configs.loader import HardwareConfig, ModelConfig
from loadgen.aggressiveness import AggressivenessSetting
from loadgen.runner import SweepConfig, run_sweep, write_csv
from loadgen.server_lifecycle import (
    ensure_port_free,
    launch_server_background,
    stop_server,
    wait_for_server_ready,
)


async def run_aggressiveness_sweep(
    model_cfg: ModelConfig,
    hw_cfg: HardwareConfig,
    settings: tuple[AggressivenessSetting, ...],
    rate_per_sec: float,
    duration_sec: float,
    slo_latency_sec: float,
    output_dir: Path,
    port: int = 8000,
    server_ready_timeout_sec: float = 400.0,
) -> list[Path]:
    base_url = f"http://localhost:{port}"
    output_paths: list[Path] = []

    for setting in settings:
        await ensure_port_free(base_url)
        log_path = output_dir / f"server_{setting.name}.log"
        process = launch_server_background(
            model_cfg, hw_cfg, setting.sink_len, setting.recent_window, port, log_path
        )
        try:
            ready = await wait_for_server_ready(base_url, timeout_sec=server_ready_timeout_sec)
            if not ready:
                raise RuntimeError(
                    f"server for aggressiveness setting '{setting.name}' did not "
                    f"become ready within {server_ready_timeout_sec}s; see {log_path}"
                )
            cfg = SweepConfig(
                base_url=base_url,
                served_model_name=model_cfg.served_model_name,
                rate_per_sec=rate_per_sec,
                duration_sec=duration_sec,
                eviction_aggressiveness=setting.name,
                slo_latency_sec=slo_latency_sec,
            )
            rows = await run_sweep(cfg)
            out_path = output_dir / f"sweep_{setting.name}.csv"
            write_csv(rows, out_path)
            output_paths.append(out_path)
        finally:
            stop_server(process)

    return output_paths
