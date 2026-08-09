from pathlib import Path

import pytest

from configs.loader import HardwareConfig, ModelConfig
from loadgen.aggressiveness import AggressivenessSetting
from loadgen import sweep_orchestrator

MODEL = ModelConfig(
    name="m", hf_repo="org/repo", served_model_name="served",
    quantization="awq", native_context_length=1000, max_context_length=1000,
)
HW = HardwareConfig(name="hw", gpu_name="gpu", vram_gb=8)
SETTINGS = (
    AggressivenessSetting("baseline", sink_len=None, recent_window=None),
    AggressivenessSetting("window_512", sink_len=64, recent_window=512),
)


class FakeProcess:
    def __init__(self, label: str):
        self.label = label


async def test_run_aggressiveness_sweep_calls_in_order_per_setting(tmp_path, monkeypatch):
    calls: list[str] = []

    async def fake_ensure_port_free(base_url):
        calls.append(f"ensure_port_free:{base_url}")

    def fake_launch(model_cfg, hw_cfg, sink_len, recent_window, port, log_path):
        calls.append(f"launch:sink={sink_len}:window={recent_window}:port={port}")
        return FakeProcess(f"sink={sink_len}")

    async def fake_wait(base_url, timeout_sec, poll_interval_sec=3.0):
        calls.append(f"wait:{base_url}")
        return True

    async def fake_run_sweep(cfg):
        calls.append(f"run_sweep:{cfg.eviction_aggressiveness}")
        return [{"eviction_aggressiveness": cfg.eviction_aggressiveness}]

    def fake_write_csv(rows, path):
        calls.append(f"write_csv:{path.name}")

    def fake_stop(process, timeout_sec=15.0):
        calls.append(f"stop:{process.label}")

    monkeypatch.setattr(sweep_orchestrator, "ensure_port_free", fake_ensure_port_free)
    monkeypatch.setattr(sweep_orchestrator, "launch_server_background", fake_launch)
    monkeypatch.setattr(sweep_orchestrator, "wait_for_server_ready", fake_wait)
    monkeypatch.setattr(sweep_orchestrator, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(sweep_orchestrator, "write_csv", fake_write_csv)
    monkeypatch.setattr(sweep_orchestrator, "stop_server", fake_stop)
    monkeypatch.setattr(sweep_orchestrator, "GPU_SETTLE_DELAY_SEC", 0.0)

    paths = await sweep_orchestrator.run_aggressiveness_sweep(
        MODEL, HW, SETTINGS, rate_per_sec=1.0, duration_sec=1.0,
        slo_latency_sec=5.0, output_dir=tmp_path, port=8000,
    )

    assert len(paths) == 2
    assert calls == [
        "ensure_port_free:http://localhost:8000",
        "launch:sink=None:window=None:port=8000",
        "wait:http://localhost:8000",
        "run_sweep:baseline",
        "write_csv:sweep_baseline.csv",
        "stop:sink=None",
        "ensure_port_free:http://localhost:8000",
        "launch:sink=64:window=512:port=8000",
        "wait:http://localhost:8000",
        "run_sweep:window_512",
        "write_csv:sweep_window_512.csv",
        "stop:sink=64",
    ]


async def test_run_aggressiveness_sweep_raises_and_stops_server_if_never_ready(tmp_path, monkeypatch):
    stopped = []

    async def fake_ensure_port_free(base_url):
        pass

    def fake_launch(model_cfg, hw_cfg, sink_len, recent_window, port, log_path):
        return FakeProcess("p")

    async def fake_wait(base_url, timeout_sec, poll_interval_sec=3.0):
        return False

    def fake_stop(process, timeout_sec=15.0):
        stopped.append(process.label)

    monkeypatch.setattr(sweep_orchestrator, "ensure_port_free", fake_ensure_port_free)
    monkeypatch.setattr(sweep_orchestrator, "launch_server_background", fake_launch)
    monkeypatch.setattr(sweep_orchestrator, "wait_for_server_ready", fake_wait)
    monkeypatch.setattr(sweep_orchestrator, "stop_server", fake_stop)
    monkeypatch.setattr(sweep_orchestrator, "GPU_SETTLE_DELAY_SEC", 0.0)

    with pytest.raises(RuntimeError, match="did not become ready"):
        await sweep_orchestrator.run_aggressiveness_sweep(
            MODEL, HW, SETTINGS[:1], rate_per_sec=1.0, duration_sec=1.0,
            slo_latency_sec=5.0, output_dir=tmp_path, port=8000,
        )
    assert stopped == ["p"]
