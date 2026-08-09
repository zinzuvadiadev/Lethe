# Aggressiveness Sweep (Milestone 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automate the full aggressiveness sweep — launch a server per
(sink_len, recent_window) setting, run the existing load generator against
it, tear it down, repeat — producing one throughput/latency CSV per setting
in `results/raw/`, ready for milestone 7's tradeoff plot.

**Architecture:** No new concepts — this composes milestone 3's load
generator (`loadgen.runner.SweepConfig`/`run_sweep`/`write_csv`, unchanged)
and milestone 4's server launcher (`serving.server.build_serve_args`,
extended by extracting its env-building logic into a reusable
`build_serve_env` function) with a new orchestration layer that manages
server subprocess lifecycle (launch in background, poll for readiness, run
the load generator, stop, move to the next setting).

**Tech Stack:** Python 3.10, `httpx` (server health polling), `subprocess`
(background server lifecycle), `pytest` + `respx` (mocking health-check
HTTP calls).

## Global Constraints

- Design source of truth: `docs/superpowers/specs/2026-08-09-kv-cache-eviction-benchmark-design.md`
  §5 (aggressiveness = recent-window size) and §12 milestone 5.
- Sweep points (confirmed with the user): sink_len fixed at 64; recent
  windows `[None (baseline, no eviction), 2048, 1024, 512, 256]` — 5
  settings total, ordered from least to most aggressive.
- Load parameters per setting: `rate=1.0`, `duration=30.0`,
  `slo_latency_sec=5.0` — identical to the milestone-3 baseline run, so
  that run's `results/raw/baseline.csv` numbers are directly comparable to
  this sweep's `baseline` setting (not strictly required to be re-run, but
  the sweep produces its own `sweep_baseline.csv` for consistency with the
  other 4 files it writes as one batch).
- **Correctness safeguard, not optional**: before launching each setting's
  server, the orchestrator must confirm nothing is already listening on the
  target port. This directly encodes a real failure mode from this same
  project's history — a stale server from an earlier milestone was still
  running and had to be manually caught and killed before milestone 4's
  live verification could proceed cleanly. A silent stale-server reuse
  here would produce 5 "different aggressiveness" CSVs that are secretly
  all the same run.
- One task = one commit.

---

### Task 1: `AggressivenessSetting` + the 5 sweep points

**Files:**
- Create: `loadgen/aggressiveness.py`
- Create: `tests/loadgen/test_aggressiveness.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `AggressivenessSetting` (fields: `name: str`, `sink_len: int | None`, `recent_window: int | None`), `DEFAULT_SETTINGS: tuple[AggressivenessSetting, ...]`. Task 4 imports `AggressivenessSetting` and `DEFAULT_SETTINGS`.

- [ ] **Step 1: Write the failing tests**

`tests/loadgen/test_aggressiveness.py`:

```python
from loadgen.aggressiveness import DEFAULT_SETTINGS, AggressivenessSetting


def test_default_settings_has_five_points_baseline_first():
    assert len(DEFAULT_SETTINGS) == 5
    assert DEFAULT_SETTINGS[0].name == "baseline"
    assert DEFAULT_SETTINGS[0].sink_len is None
    assert DEFAULT_SETTINGS[0].recent_window is None


def test_default_settings_eviction_points_share_sink_len_64():
    eviction_points = DEFAULT_SETTINGS[1:]
    assert all(s.sink_len == 64 for s in eviction_points)


def test_default_settings_windows_descend_in_aggressiveness_order():
    windows = [s.recent_window for s in DEFAULT_SETTINGS[1:]]
    assert windows == [2048, 1024, 512, 256]


def test_default_settings_names_are_unique():
    names = [s.name for s in DEFAULT_SETTINGS]
    assert len(names) == len(set(names))


def test_aggressiveness_setting_is_a_plain_dataclass_with_expected_fields():
    s = AggressivenessSetting(name="x", sink_len=1, recent_window=2)
    assert (s.name, s.sink_len, s.recent_window) == ("x", 1, 2)
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/loadgen/test_aggressiveness.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'loadgen.aggressiveness'`.

- [ ] **Step 3: Write `loadgen/aggressiveness.py`**

```python
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class AggressivenessSetting:
    name: str
    sink_len: int | None
    recent_window: int | None


# See docs/superpowers/specs/2026-08-09-kv-cache-eviction-benchmark-design.md
# §5: aggressiveness = recent-window size. sink_len fixed at 64 (the value
# already validated live in milestone 4) across all eviction points; only
# the window shrinks as aggressiveness increases.
DEFAULT_SETTINGS: tuple[AggressivenessSetting, ...] = (
    AggressivenessSetting("baseline", sink_len=None, recent_window=None),
    AggressivenessSetting("window_2048", sink_len=64, recent_window=2048),
    AggressivenessSetting("window_1024", sink_len=64, recent_window=1024),
    AggressivenessSetting("window_512", sink_len=64, recent_window=512),
    AggressivenessSetting("window_256", sink_len=64, recent_window=256),
)
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/pytest tests/loadgen/test_aggressiveness.py -v
```

Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add loadgen/aggressiveness.py tests/loadgen/test_aggressiveness.py
git commit -m "Add the 5-point aggressiveness sweep settings (sink_len=64, windows 2048-256)"
```

---

### Task 2: Extract `build_serve_env` from `serving/server.py`

**Files:**
- Modify: `serving/server.py`
- Modify: `tests/serving/test_server.py`

**Interfaces:**
- Consumes: `find_pip_cuda_home` (existing, unchanged).
- Produces: `build_serve_env(sink_len: int | None, recent_window: int | None, base_env: dict[str, str] | None = None) -> dict[str, str]`. Task 3 imports this to launch background servers without duplicating `main()`'s env-building logic.

This is a pure refactor — `main()`'s behavior must not change. The exact
env-building logic currently inline in `main()` (read the file first) moves
into a new function; `main()` calls it instead of repeating it.

- [ ] **Step 1: Write the failing test**

Add to `tests/serving/test_server.py`:

```python
def test_build_serve_env_sets_pythonpath_to_repo_root():
    env = build_serve_env(sink_len=None, recent_window=None, base_env={})
    assert str(Path(__file__).resolve().parents[2]) in env["PYTHONPATH"]


def test_build_serve_env_sets_flashinfer_sampler_disabled_by_default():
    env = build_serve_env(sink_len=None, recent_window=None, base_env={})
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_build_serve_env_respects_existing_flashinfer_sampler_setting():
    env = build_serve_env(
        sink_len=None, recent_window=None,
        base_env={"VLLM_USE_FLASHINFER_SAMPLER": "1"},
    )
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "1"


def test_build_serve_env_omits_eviction_env_vars_when_disabled():
    env = build_serve_env(sink_len=None, recent_window=None, base_env={})
    assert "LETHE_SINK_LEN" not in env
    assert "LETHE_RSWA_WINDOW" not in env


def test_build_serve_env_sets_eviction_env_vars_when_enabled():
    env = build_serve_env(sink_len=64, recent_window=512, base_env={})
    assert env["LETHE_SINK_LEN"] == "64"
    assert env["LETHE_RSWA_WINDOW"] == "512"
```

Add `build_serve_env` to the existing import line at the top of the test file:

```python
from serving.server import build_serve_args, build_serve_env, find_pip_cuda_home
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/serving/test_server.py -v -k build_serve_env
```

Expected: FAIL — `ImportError: cannot import name 'build_serve_env'`.

- [ ] **Step 3: Extract `build_serve_env` in `serving/server.py`**

Add this function after `build_serve_args` and before `main`:

```python
def build_serve_env(
    sink_len: int | None,
    recent_window: int | None,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    venv_bin = Path(sys.executable).parent
    env = dict(base_env) if base_env is not None else os.environ.copy()
    # .venv/bin also holds pip-installed build tools (ninja) that vLLM's
    # runtime JIT kernel compilation (flashinfer) shells out to by bare name;
    # without it on PATH those subprocess lookups fail with FileNotFoundError.
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    # repo root on PYTHONPATH: --model-class-overrides/--scheduler-cls
    # resolve "serving.eviction.*:ClassName" by lazy import INSIDE this
    # subprocess, which needs the repo root importable, not just our own
    # (parent) process's sys.path.
    repo_root = Path(__file__).resolve().parent.parent
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    site_packages = venv_bin.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    cuda_home = find_pip_cuda_home(site_packages)
    if cuda_home is not None and "CUDA_HOME" not in env:
        env["CUDA_HOME"] = str(cuda_home)
        env["PATH"] = f"{cuda_home / 'bin'}{os.pathsep}{env['PATH']}"

    # FlashInfer's JIT-compiled top-k/top-p sampling kernel fails to build on
    # this GPU/CUDA-toolkit combination: its bundled CCCL headers reject the
    # pip-installed CUDA 13.3 nvcc + sm_120f (Blackwell) target with "CUDA
    # compiler and CUDA toolkit headers are incompatible". This falls back to
    # vLLM's PyTorch-native sampler, which doesn't hit that JIT path.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    if sink_len is not None:
        env["LETHE_SINK_LEN"] = str(sink_len)
    if recent_window is not None:
        env["LETHE_RSWA_WINDOW"] = str(recent_window)
    return env
```

Replace `main()`'s inline env-building block (everything from `env =
os.environ.copy()` through the `LETHE_RSWA_WINDOW` lines) with a single call:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a config-driven vLLM server")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--sink-len", type=int, default=None,
        help="Enable eviction: protect this many leading tokens (requires --recent-window too)",
    )
    parser.add_argument(
        "--recent-window", type=int, default=None,
        help="Enable eviction: protect this many trailing tokens (requires --sink-len too)",
    )
    args = parser.parse_args()

    model_cfg = load_model_config(args.model_config)
    hw_cfg = load_hardware_config(args.hardware_config)
    argv = build_serve_args(
        model_cfg, hw_cfg, port=args.port,
        sink_len=args.sink_len, recent_window=args.recent_window,
    )
    venv_bin = Path(sys.executable).parent
    # Resolve the console-script path explicitly: subprocess.run inherits our
    # PATH, which won't include .venv/bin unless the venv was activated (we
    # invoke .venv/bin/python directly instead), so a bare "vllm" lookup fails.
    argv[0] = str(venv_bin / "vllm")

    env = build_serve_env(args.sink_len, args.recent_window)
    print(f"launching: {' '.join(argv)}")
    subprocess.run(argv, check=True, env=env)
```

Note: the `find_pip_cuda_home` print statement ("CUDA_HOME not set; using
pip-installed CUDA toolkit at ...") that existed inline in the old `main()`
is dropped in this refactor — `build_serve_env` doesn't print (it's meant
to be reused by the background-launching orchestrator in Task 3, where a
print to stdout would land in the server's own log file, not the caller's,
and be lost). This is a minor diagnostic-message loss, not a behavior
change — `CUDA_HOME` itself is still set in the returned env dict exactly
as before.

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/pytest tests/serving/test_server.py -v
```

Expected: PASS (all tests, existing + 5 new).

- [ ] **Step 5: Run the full test suite**

```bash
.venv/bin/pytest -v
```

Expected: all existing tests still pass (this was a refactor of tested,
working code — a regression here means the extraction changed behavior).

- [ ] **Step 6: Commit**

```bash
git add serving/server.py tests/serving/test_server.py
git commit -m "Extract build_serve_env from serving/server.py for reuse by the sweep orchestrator"
```

---

### Task 3: Server lifecycle helpers — launch in background, wait for ready, ensure port free, stop

**Files:**
- Create: `loadgen/server_lifecycle.py`
- Create: `tests/loadgen/test_server_lifecycle.py`

**Interfaces:**
- Consumes: `serving.server.build_serve_args`, `serving.server.build_serve_env` (Task 2).
- Produces: `launch_server_background(model_cfg, hw_cfg, sink_len, recent_window, port, log_path) -> subprocess.Popen`, `async wait_for_server_ready(base_url, timeout_sec=400.0, poll_interval_sec=3.0) -> bool`, `async ensure_port_free(base_url) -> None` (raises `RuntimeError` if something is already healthy there), `stop_server(process, timeout_sec=15.0) -> None`. Task 4 imports all four.

- [ ] **Step 1: Write the failing tests for the mockable async helpers**

`tests/loadgen/test_server_lifecycle.py`:

```python
import httpx
import pytest
import respx

from loadgen.server_lifecycle import ensure_port_free, wait_for_server_ready


@respx.mock
async def test_wait_for_server_ready_returns_true_on_first_healthy_response():
    respx.get("http://localhost:8000/health").mock(return_value=httpx.Response(200))
    result = await wait_for_server_ready("http://localhost:8000", timeout_sec=5.0, poll_interval_sec=0.01)
    assert result is True


@respx.mock
async def test_wait_for_server_ready_times_out_if_never_healthy():
    respx.get("http://localhost:8000/health").mock(side_effect=httpx.ConnectError("refused"))
    result = await wait_for_server_ready("http://localhost:8000", timeout_sec=0.05, poll_interval_sec=0.01)
    assert result is False


@respx.mock
async def test_wait_for_server_ready_recovers_after_transient_failures():
    route = respx.get("http://localhost:8000/health")
    route.side_effect = [
        httpx.ConnectError("refused"),
        httpx.ConnectError("refused"),
        httpx.Response(200),
    ]
    result = await wait_for_server_ready("http://localhost:8000", timeout_sec=5.0, poll_interval_sec=0.01)
    assert result is True


@respx.mock
async def test_ensure_port_free_passes_when_nothing_listening():
    respx.get("http://localhost:8000/health").mock(side_effect=httpx.ConnectError("refused"))
    await ensure_port_free("http://localhost:8000")  # must not raise


@respx.mock
async def test_ensure_port_free_raises_when_something_healthy():
    respx.get("http://localhost:8000/health").mock(return_value=httpx.Response(200))
    with pytest.raises(RuntimeError, match="already responding"):
        await ensure_port_free("http://localhost:8000")
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/loadgen/test_server_lifecycle.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'loadgen.server_lifecycle'`.

- [ ] **Step 3: Write `loadgen/server_lifecycle.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/pytest tests/loadgen/test_server_lifecycle.py -v
```

Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add loadgen/server_lifecycle.py tests/loadgen/test_server_lifecycle.py
git commit -m "Add server lifecycle helpers: background launch, readiness polling, port-conflict guard"
```

---

### Task 4: Sweep orchestrator + CLI

**Files:**
- Create: `loadgen/sweep_orchestrator.py`
- Create: `loadgen/sweep_cli.py`
- Test: `tests/loadgen/test_sweep_orchestrator.py`

**Interfaces:**
- Consumes: `loadgen.aggressiveness.AggressivenessSetting`, `loadgen.aggressiveness.DEFAULT_SETTINGS` (Task 1); `loadgen.server_lifecycle.launch_server_background`, `wait_for_server_ready`, `ensure_port_free`, `stop_server` (Task 3); `loadgen.runner.SweepConfig`, `run_sweep`, `write_csv` (existing, unchanged); `configs.loader.load_model_config`, `load_hardware_config` (existing).
- Produces: `async run_aggressiveness_sweep(model_cfg, hw_cfg, settings, rate_per_sec, duration_sec, slo_latency_sec, output_dir, port=8000, server_ready_timeout_sec=400.0) -> list[Path]`. This is the last task in this plan.

- [ ] **Step 1: Write the failing orchestration test**

This tests sequencing (launch → wait → sweep → write → stop, per setting,
in order) by monkeypatching the four I/O primitives from
`loadgen.server_lifecycle` and `loadgen.runner.run_sweep` — it does not
touch real subprocesses or network, matching how `loadgen/server_lifecycle.py`
itself was tested against mocked HTTP in Task 3.

`tests/loadgen/test_sweep_orchestrator.py`:

```python
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

    with pytest.raises(RuntimeError, match="did not become ready"):
        await sweep_orchestrator.run_aggressiveness_sweep(
            MODEL, HW, SETTINGS[:1], rate_per_sec=1.0, duration_sec=1.0,
            slo_latency_sec=5.0, output_dir=tmp_path, port=8000,
        )
    assert stopped == ["p"]
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/loadgen/test_sweep_orchestrator.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'loadgen.sweep_orchestrator'`.

- [ ] **Step 3: Write `loadgen/sweep_orchestrator.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/pytest tests/loadgen/test_sweep_orchestrator.py -v
```

Expected: PASS (2 passed).

- [ ] **Step 5: Write the CLI entrypoint**

`loadgen/sweep_cli.py`:

```python
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
```

- [ ] **Step 6: Run the full test suite**

```bash
.venv/bin/pytest -v
```

Expected: all tests pass (previous total + this plan's ~17 new tests).

- [ ] **Step 7: Commit**

```bash
git add loadgen/sweep_orchestrator.py loadgen/sweep_cli.py tests/loadgen/test_sweep_orchestrator.py
git commit -m "Add aggressiveness sweep orchestrator and CLI"
```

---

### Task 5: Update the dashboard milestone marker

**Files:**
- Modify: `dashboard/milestones.py`

- [ ] **Step 1: Update the milestone 5 marker path**

Change:

```python
    Milestone(5, "Sweep across aggressiveness", "results/raw/aggressiveness_sweep.csv"),
```

to:

```python
    Milestone(5, "Sweep across aggressiveness", "results/raw/sweep_window_256.csv"),
```

(The last file the real sweep produces — see Task 6 — makes a reasonable
"did the full sweep actually complete" signal, the same reasoning already
used for milestone 4's marker.)

- [ ] **Step 2: Regenerate the dashboard and confirm it still runs cleanly**

```bash
.venv/bin/python -m dashboard.cli
```

Expected: succeeds (milestone 5 will show pending until Task 6 actually
produces the file).

- [ ] **Step 3: Commit**

```bash
git add dashboard/milestones.py
git commit -m "Update milestone 5 marker to the real sweep output filename"
```

---

### Task 6: Live verification — run the real 5-point sweep

No pre-written code — this is a real run against the live GPU, the same
pattern as milestones 2 and 4's live verification tasks.

- [ ] **Step 1: Confirm no server is currently running**

```bash
ps aux | grep "vllm serve" | grep -v grep
```

If anything is running, stop it first (the sweep's own `ensure_port_free`
guard from Task 3 will raise a clear error rather than silently reusing a
stale server, but starting clean avoids wasting a sweep point on an
avoidable failure).

- [ ] **Step 2: Run the sweep**

```bash
.venv/bin/python -m loadgen.sweep_cli \
  --model-config configs/models/qwen3-4b-instruct-2507-awq.yaml \
  --hardware-config configs/hardware/rtx5070-laptop.yaml \
  --rate 1.0 --duration 30 --slo-latency-sec 5.0 \
  --out-dir results/raw
```

Expected total wall-clock time: roughly 5-10 minutes (5 settings × ~40-90s
server startup + 30s load-gen + teardown each; the `baseline` setting uses
a different model class than the 4 eviction settings and will need a fresh
`torch.compile`, so expect it to take longer than the eviction points,
which likely share a warm compile cache — this is expected, not a bug).
Watch for `wrote 5 sweep files:` at the end, listing
`results/raw/sweep_baseline.csv` through `results/raw/sweep_window_256.csv`.

If any setting's server fails to become ready, the error message names
which `server_<setting>.log` to check — follow the same real-error
diagnosis process used in every prior live-verification task in this
project (read the actual vLLM error, don't guess a fix).

- [ ] **Step 3: Sanity-check the results**

For each of the 5 CSVs, confirm: non-trivial row count (~30, matching
`rate=1.0 × duration=30s`), success rate is high (some SLO misses are
expected and fine — that's real data, not a bug), and `eviction_aggressiveness`
matches the filename's setting name throughout.

```bash
for f in results/raw/sweep_*.csv; do
  echo "== $f =="
  wc -l "$f"
  .venv/bin/python -c "
import pandas as pd
df = pd.read_csv('$f')
print('success rate:', df['success'].mean())
print('mean latency:', df['latency_sec'].mean())
"
done
```

- [ ] **Step 4: Regenerate the dashboard and confirm milestone 5 flips to done**

```bash
.venv/bin/python -m dashboard.cli
grep -o 'milestone-[a-z]*">\[.\] Sweep across aggressiveness' results/dashboard.html
```

Expected: `milestone-done">[x] Sweep across aggressiveness`.

- [ ] **Step 5: Commit the real sweep output**

```bash
git add results/raw/sweep_baseline.csv results/raw/sweep_window_2048.csv results/raw/sweep_window_1024.csv results/raw/sweep_window_512.csv results/raw/sweep_window_256.csv
git commit -m "Add real 5-point aggressiveness sweep results"
```

(Server `.log` files under `results/raw/` are gitignored via the existing
`*.log` rule — don't force-add them.)

---

## After this plan

Milestone 5 is done: `results/raw/sweep_*.csv` holds real throughput/latency
data across 5 aggressiveness settings. The next plan covers milestone 6
(LongBench subset eval harness) — a new `/eval/` package measuring the
quality side of the tradeoff this sweep's throughput data will eventually
be plotted against in milestone 7. Milestone 6 involves real design
decisions (which LongBench tasks fit this deployment's context budget,
how perplexity + task accuracy get scored per aggressiveness setting) not
yet pinned down — that plan should go through a brainstorming/design pass
before writing tasks, unlike this mechanical one.
