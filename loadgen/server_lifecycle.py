from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import httpx

from configs.loader import HardwareConfig, ModelConfig
from serving.server import build_serve_args, build_serve_env


async def ensure_port_free(base_url: str) -> None:
    """Refuse to proceed if a server is already healthy at base_url.

    Real precedent for why this guard exists, not a hypothetical: during
    this project's milestone 4 live verification, a stale server from an
    earlier milestone was still running on the same port and had to be
    manually caught and killed. A silent stale-server reuse during an
    automated multi-setting sweep would produce N "different aggressiveness"
    CSVs that are secretly all the same run.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{base_url}/health", timeout=2.0)
        except httpx.HTTPError:
            return
        if response.status_code == 200:
            raise RuntimeError(
                f"A server is already responding at {base_url} — stop it before "
                "running the sweep, or it will silently produce identical "
                "results for every aggressiveness setting."
            )


async def wait_for_server_ready(
    base_url: str,
    timeout_sec: float = 400.0,
    poll_interval_sec: float = 3.0,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{base_url}/health", timeout=5.0)
                if response.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(poll_interval_sec)
    return False


def launch_server_background(
    model_cfg: ModelConfig,
    hw_cfg: HardwareConfig,
    sink_len: int | None,
    recent_window: int | None,
    port: int,
    log_path: Path,
) -> subprocess.Popen:
    argv = build_serve_args(
        model_cfg, hw_cfg, port=port, sink_len=sink_len, recent_window=recent_window
    )
    venv_bin = Path(sys.executable).parent
    argv[0] = str(venv_bin / "vllm")
    env = build_serve_env(sink_len, recent_window)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    return subprocess.Popen(argv, env=env, stdout=log_file, stderr=subprocess.STDOUT)


def stop_server(process: subprocess.Popen, timeout_sec: float = 15.0) -> None:
    process.terminate()
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
