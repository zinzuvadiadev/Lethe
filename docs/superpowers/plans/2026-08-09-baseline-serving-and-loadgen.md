# Baseline vLLM Serving + Load Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a config-driven vLLM server for Qwen3-4B-Instruct-2507 (AWQ) on the RTX 5070, and a load generator that drives it with Poisson-arrival concurrent traffic and records per-request throughput/latency to CSV — producing the first real baseline numbers.

**Architecture:** A shared `configs/` package holds model and hardware definitions as YAML plus a loader that returns typed dataclasses; `serving/` reads those configs to launch vLLM with no hardcoded model/GPU assumptions; `loadgen/` is an async httpx-based client that samples request shapes, schedules Poisson arrivals against the running server, and writes results to CSV. Every later milestone (eviction policy, sweeps, eval, 4090 phase) builds on these three packages without modifying them structurally.

**Tech Stack:** Python 3.10, vLLM (OpenAI-compatible server), httpx (async client) + respx (test mocking), PyYAML, pytest + pytest-asyncio, pandas/matplotlib (installed now, used starting in the next plan).

## Global Constraints

- Design source of truth: `docs/superpowers/specs/2026-08-09-kv-cache-eviction-benchmark-design.md`.
- Model: Qwen3-4B-Instruct-2507, AWQ 4-bit, community quant `cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit` (verified to exist; the official Qwen repo has no AWQ variant).
- Hardware for this plan: RTX 5070 Laptop, 8GB VRAM only. The RTX 4090 is out of scope here — that's design-doc milestone 9, a separate future plan.
- No hardcoded model name, quantization scheme, or GPU assumption in `serving/`, `loadgen/`, or (later) `eval/` code — everything reads from `configs/models/*.yaml` and `configs/hardware/*.yaml` via `configs/loader.py`.
- One task = one commit. Do not squash multiple tasks into one commit.
- `.gitignore` excludes model weights, HF cache, venvs, `__pycache__`, and logs. It does **not** exclude `results/raw/*.csv` — raw benchmark output is a tracked deliverable, not a build artifact.
- This plan covers design-doc milestones 1-3 only (scaffold, config-driven baseline serving, load generator + baseline throughput numbers). Milestones 4-9 (eviction policy, aggressiveness sweeps, LongBench eval, results/plot, README, RTX 4090 phase) get their own plan(s), written once this plan's outputs — the actual installed vLLM version's internals, and real baseline throughput numbers — are known. Writing that plan now would require guessing vLLM internals we haven't inspected yet.

---

### Task 1: Repository scaffold

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `README.md`
- Create: `configs/__init__.py`, `serving/__init__.py`, `loadgen/__init__.py`, `eval/__init__.py`
- Create: `tests/__init__.py`, `tests/configs/__init__.py`, `tests/serving/__init__.py`, `tests/loadgen/__init__.py`
- Create: `results/raw/.gitkeep`

**Interfaces:**
- Produces: the package layout every later task imports from (`configs`, `serving`, `loadgen`, `eval`), and the pytest config (`pytest.ini` with `asyncio_mode = auto`) that Task 6's async tests rely on.

- [ ] **Step 1: Create the directory structure**

```bash
mkdir -p configs/models configs/hardware
mkdir -p serving loadgen eval
mkdir -p results/raw
mkdir -p tests/configs tests/serving tests/loadgen
```

- [ ] **Step 2: Create empty `__init__.py` files so these are importable packages**

```bash
touch configs/__init__.py serving/__init__.py loadgen/__init__.py eval/__init__.py
touch tests/__init__.py tests/configs/__init__.py tests/serving/__init__.py tests/loadgen/__init__.py
touch results/raw/.gitkeep
```

- [ ] **Step 3: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.env

# model weights / HF cache — never commit
models/
.cache/
*.safetensors
*.bin
*.gguf

# vllm / server logs
*.log
nohup.out
```

- [ ] **Step 4: Write `requirements.txt`**

```
pyyaml>=6.0
pytest>=8.0
pytest-asyncio>=0.24
httpx>=0.27
respx>=0.21
numpy>=1.26
pandas>=2.2
matplotlib>=3.9
```

- [ ] **Step 5: Write `pytest.ini`**

```ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 6: Write `README.md` skeleton**

```markdown
# KV-Cache Eviction Policy + Throughput/Quality Benchmark

## Problem

_Filled in at milestone 8 — see `docs/superpowers/specs/2026-08-09-kv-cache-eviction-benchmark-design.md` for the full design in the meantime._

## Method

_Filled in at milestone 8._

## Hardware & Models

_Filled in at milestone 8._

## Reproducing

_Filled in at milestone 8._

## Results

_Filled in at milestones 7-8._

## Limitations

_Filled in at milestone 8._
```

- [ ] **Step 7: Create a venv and install dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: installs cleanly, no errors.

- [ ] **Step 8: Verify the scaffold and that pytest runs cleanly with zero tests**

```bash
source .venv/bin/activate
find . -not -path './.git*' -not -path './.venv*' -not -path './docs/*' | sort
pytest
```

Expected: the `find` output matches the directories/files created above; `pytest` reports `no tests ran` (exit code 5) without import errors.

- [ ] **Step 9: Commit**

```bash
git add .gitignore requirements.txt pytest.ini README.md configs serving loadgen eval tests results
git commit -m "Scaffold repo structure, gitignore, and dependency baseline"
```

---

### Task 2: Config module — model & hardware YAML + typed loader

**Files:**
- Create: `configs/models/qwen3-4b-instruct-2507-awq.yaml`
- Create: `configs/hardware/rtx5070-laptop.yaml`
- Create: `configs/loader.py`
- Test: `tests/configs/test_loader.py`

**Interfaces:**
- Consumes: nothing (first real module).
- Produces: `ModelConfig` (fields: `name: str`, `hf_repo: str`, `served_model_name: str`, `quantization: str`, `native_context_length: int`, `max_context_length: int`), `HardwareConfig` (fields: `name: str`, `gpu_name: str`, `vram_gb: float`), `load_model_config(path: str | Path) -> ModelConfig`, `load_hardware_config(path: str | Path) -> HardwareConfig`. Every later task that needs model/hardware info imports these two functions and two dataclasses from `configs.loader`.

- [ ] **Step 1: Write the failing tests**

`tests/configs/test_loader.py`:

```python
from pathlib import Path

import pytest

from configs.loader import (
    HardwareConfig,
    ModelConfig,
    load_hardware_config,
    load_model_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_model_config_reads_qwen3_4b_awq():
    cfg = load_model_config(REPO_ROOT / "configs/models/qwen3-4b-instruct-2507-awq.yaml")
    assert cfg == ModelConfig(
        name="qwen3-4b-instruct-2507-awq",
        hf_repo="cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit",
        served_model_name="qwen3-4b-instruct-2507",
        quantization="awq",
        native_context_length=262144,
        max_context_length=32768,
    )


def test_load_hardware_config_reads_rtx5070():
    cfg = load_hardware_config(REPO_ROOT / "configs/hardware/rtx5070-laptop.yaml")
    assert cfg == HardwareConfig(
        name="rtx5070-laptop",
        gpu_name="NVIDIA GeForce RTX 5070 Laptop GPU",
        vram_gb=8,
    )


def test_load_model_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_model_config(REPO_ROOT / "configs/models/does-not-exist.yaml")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
source .venv/bin/activate
pytest tests/configs/test_loader.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'configs.loader'` (and the two YAML files don't exist yet either).

- [ ] **Step 3: Write the YAML config files**

`configs/models/qwen3-4b-instruct-2507-awq.yaml`:

```yaml
name: qwen3-4b-instruct-2507-awq
hf_repo: cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit
served_model_name: qwen3-4b-instruct-2507
quantization: awq
native_context_length: 262144
max_context_length: 32768
```

`configs/hardware/rtx5070-laptop.yaml`:

```yaml
name: rtx5070-laptop
gpu_name: NVIDIA GeForce RTX 5070 Laptop GPU
vram_gb: 8
```

- [ ] **Step 4: Write the loader**

`configs/loader.py`:

```python
from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    name: str
    hf_repo: str
    served_model_name: str
    quantization: str
    native_context_length: int
    max_context_length: int


@dataclasses.dataclass(frozen=True)
class HardwareConfig:
    name: str
    gpu_name: str
    vram_gb: float


def load_model_config(path: str | Path) -> ModelConfig:
    data = yaml.safe_load(Path(path).read_text())
    return ModelConfig(**data)


def load_hardware_config(path: str | Path) -> HardwareConfig:
    data = yaml.safe_load(Path(path).read_text())
    return HardwareConfig(**data)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
source .venv/bin/activate
pytest tests/configs/test_loader.py -v
```

Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add configs/models configs/hardware configs/loader.py tests/configs/test_loader.py
git commit -m "Add config-driven model/hardware definitions and typed loader"
```

---

### Task 3: GPU + vLLM install verification

**Files:**
- Create: `serving/verify_gpu.py`
- Modify: `requirements.txt` (append the pinned `vllm` version once installed)

**Interfaces:**
- Consumes: nothing.
- Produces: confirmation (not code other tasks import) that `torch` sees the RTX 5070 at the right compute capability and that `vllm` imports cleanly, plus the pinned `vllm==<version>` line in `requirements.txt` that Task 4 depends on being installed.

This task is a verification spike, not unit-testable — it's confirming facts about the installed environment, not implementing logic. Run it for real and record what you observe.

- [ ] **Step 1: Write the GPU verification script**

`serving/verify_gpu.py`:

```python
from __future__ import annotations

import torch


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"device: {name}")
    print(f"capability: {capability}")
    arch_list = torch.cuda.get_arch_list()
    print(f"arch list: {arch_list}")
    sm = f"sm_{capability[0]}{capability[1]}"
    if sm not in arch_list:
        raise SystemExit(f"{sm} not in torch arch list {arch_list}")
    print(f"OK: {sm} supported by installed torch")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and confirm torch itself is fine on this GPU**

```bash
source .venv/bin/activate
python serving/verify_gpu.py
```

Expected: prints `capability: (12, 0)`, `sm_120` present in the arch list, and ends with `OK: sm_120 supported by installed torch`. (This was already confirmed manually during design — torch `2.12.0+cu130` here already lists `sm_120`. This step re-confirms it in the venv, since a fresh venv reinstalling torch as a transitive dependency of vllm could pull a different build.)

- [ ] **Step 3: Install vLLM and verify it imports**

```bash
source .venv/bin/activate
pip install vllm
python -c "import vllm; print(vllm.__version__)"
```

Expected: installs and imports without error, prints a version string.

- [ ] **Step 4: If the import fails with an sm_120/kernel-not-found error, apply the documented Blackwell workaround**

This is a known risk for this GPU generation (see design doc §12), not a hypothetical — check for it explicitly. If Step 3's import (or a subsequent real server launch in Task 4) raises a CUDA kernel image / "no kernel image available" error referencing compute capability 12.0:

```bash
source .venv/bin/activate
pip uninstall -y vllm
TORCH_CUDA_ARCH_LIST="12.0+PTX" pip install --no-cache-dir --force-reinstall vllm
python -c "import vllm; print(vllm.__version__)"
```

If that still fails, record the exact error in a new "Setup notes" subsection under `## Hardware & Models` in `README.md` before moving on — later milestones need to know whether we're on a patched install.

- [ ] **Step 5: Pin the working version in `requirements.txt`**

```bash
source .venv/bin/activate
pip show vllm | grep Version
```

Append the exact result to `requirements.txt`, e.g. replace the plan's implicit "installed manually in Task 3" state with an explicit pin:

```
vllm==<version printed above>
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt serving/verify_gpu.py
git commit -m "Verify RTX 5070 (sm_120) GPU + vLLM install, pin vLLM version"
```

---

### Task 4: Config-driven vLLM server launcher + smoke test

**Files:**
- Create: `serving/server.py`
- Create: `serving/smoke_test.py`
- Test: `tests/serving/test_server.py`

**Interfaces:**
- Consumes: `configs.loader.ModelConfig`, `configs.loader.HardwareConfig`, `configs.loader.load_model_config`, `configs.loader.load_hardware_config` (Task 2).
- Produces: `build_serve_args(model_cfg: ModelConfig, hw_cfg: HardwareConfig, port: int = 8000) -> list[str]` — a pure function later tasks/tests can reuse if they need to launch or reason about server args without touching subprocess.

- [ ] **Step 1: Write the failing test for the pure arg-building function**

`tests/serving/test_server.py`:

```python
from configs.loader import HardwareConfig, ModelConfig
from serving.server import build_serve_args

MODEL = ModelConfig(
    name="qwen3-4b-instruct-2507-awq",
    hf_repo="cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit",
    served_model_name="qwen3-4b-instruct-2507",
    quantization="awq",
    native_context_length=262144,
    max_context_length=32768,
)
SMALL_HW = HardwareConfig(name="rtx5070-laptop", gpu_name="RTX 5070 Laptop", vram_gb=8)
BIG_HW = HardwareConfig(name="rtx4090", gpu_name="RTX 4090", vram_gb=24)


def test_build_serve_args_includes_model_and_quantization():
    args = build_serve_args(MODEL, SMALL_HW, port=8000)
    assert args[:3] == ["vllm", "serve", "cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit"]
    assert args[args.index("--quantization") + 1] == "awq"
    assert args[args.index("--max-model-len") + 1] == "32768"
    assert args[args.index("--port") + 1] == "8000"
    assert args[args.index("--served-model-name") + 1] == "qwen3-4b-instruct-2507"


def test_build_serve_args_lowers_gpu_util_for_small_vram():
    args = build_serve_args(MODEL, SMALL_HW)
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.85"


def test_build_serve_args_raises_gpu_util_for_large_vram():
    args = build_serve_args(MODEL, BIG_HW)
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.9"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
source .venv/bin/activate
pytest tests/serving/test_server.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'serving.server'`.

- [ ] **Step 3: Write `serving/server.py`**

```python
from __future__ import annotations

import argparse
import subprocess

from configs.loader import HardwareConfig, ModelConfig, load_hardware_config, load_model_config


def build_serve_args(model_cfg: ModelConfig, hw_cfg: HardwareConfig, port: int = 8000) -> list[str]:
    gpu_util = 0.85 if hw_cfg.vram_gb <= 12 else 0.9
    return [
        "vllm",
        "serve",
        model_cfg.hf_repo,
        "--served-model-name", model_cfg.served_model_name,
        "--quantization", model_cfg.quantization,
        "--max-model-len", str(model_cfg.max_context_length),
        "--gpu-memory-utilization", str(gpu_util),
        "--port", str(port),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a config-driven vLLM server")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model_cfg = load_model_config(args.model_config)
    hw_cfg = load_hardware_config(args.hardware_config)
    argv = build_serve_args(model_cfg, hw_cfg, port=args.port)
    print(f"launching: {' '.join(argv)}")
    subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
source .venv/bin/activate
pytest tests/serving/test_server.py -v
```

Expected: PASS (3 passed).

- [ ] **Step 5: Write the smoke test script**

`serving/smoke_test.py`:

```python
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
```

- [ ] **Step 6: Launch the real server and run the smoke test against it**

In one terminal:

```bash
source .venv/bin/activate
python serving/server.py --model-config configs/models/qwen3-4b-instruct-2507-awq.yaml --hardware-config configs/hardware/rtx5070-laptop.yaml
```

Wait for the log line indicating the server is ready (`Uvicorn running on http://0.0.0.0:8000` or equivalent). In a second terminal:

```bash
source .venv/bin/activate
python serving/smoke_test.py
```

Expected: `status: 200`, a non-empty completion printed, ends with `OK: server responded with a non-empty completion`. If the model download hasn't happened yet, this step will first pull ~2.5GB from Hugging Face — expected, not an error.

- [ ] **Step 7: Commit**

```bash
git add serving/server.py serving/smoke_test.py tests/serving/test_server.py
git commit -m "Add config-driven vLLM server launcher and smoke test"
```

---

### Task 5: Load generator sampling — context length, output length, Poisson arrivals

**Files:**
- Create: `loadgen/sampling.py`
- Create: `loadgen/arrivals.py`
- Test: `tests/loadgen/test_sampling.py`
- Test: `tests/loadgen/test_arrivals.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ContextLengthBucket` (fields: `name: str`, `min_tokens: int`, `max_tokens: int`, `weight: float`), `DEFAULT_BUCKETS: tuple[ContextLengthBucket, ...]`, `sample_context_length(buckets=DEFAULT_BUCKETS, rng=None) -> int`, `sample_output_length(min_tokens=32, max_tokens=512, rng=None) -> int`, `poisson_arrival_times(rate_per_sec: float, duration_sec: float, rng=None) -> list[float]`. Task 7's runner imports all of these directly.

- [ ] **Step 1: Write the failing sampling tests**

`tests/loadgen/test_sampling.py`:

```python
import random

from loadgen.sampling import ContextLengthBucket, sample_context_length, sample_output_length


def test_sample_context_length_within_bucket_bounds():
    rng = random.Random(0)
    buckets = (ContextLengthBucket("only", 100, 200, 1.0),)
    for _ in range(200):
        value = sample_context_length(buckets, rng=rng)
        assert 100 <= value <= 200


def test_sample_context_length_respects_bucket_weights():
    rng = random.Random(42)
    buckets = (
        ContextLengthBucket("a", 0, 10, 0.9),
        ContextLengthBucket("b", 1000, 1010, 0.1),
    )
    samples = [sample_context_length(buckets, rng=rng) for _ in range(2000)]
    fraction_low = sum(1 for s in samples if s <= 10) / len(samples)
    assert 0.83 <= fraction_low <= 0.97


def test_sample_output_length_within_bounds():
    rng = random.Random(1)
    for _ in range(200):
        value = sample_output_length(32, 64, rng=rng)
        assert 32 <= value <= 64
```

- [ ] **Step 2: Run to verify failure**

```bash
source .venv/bin/activate
pytest tests/loadgen/test_sampling.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'loadgen.sampling'`.

- [ ] **Step 3: Write `loadgen/sampling.py`**

```python
from __future__ import annotations

import dataclasses
import random


@dataclasses.dataclass(frozen=True)
class ContextLengthBucket:
    name: str
    min_tokens: int
    max_tokens: int
    weight: float


DEFAULT_BUCKETS: tuple[ContextLengthBucket, ...] = (
    ContextLengthBucket("short", 128, 1024, 0.5),
    ContextLengthBucket("medium", 1024, 8192, 0.35),
    ContextLengthBucket("long", 8192, 32768, 0.15),
)


def sample_context_length(
    buckets: tuple[ContextLengthBucket, ...] = DEFAULT_BUCKETS,
    rng: random.Random | None = None,
) -> int:
    rng = rng or random.Random()
    bucket = rng.choices(buckets, weights=[b.weight for b in buckets], k=1)[0]
    return rng.randint(bucket.min_tokens, bucket.max_tokens)


def sample_output_length(
    min_tokens: int = 32,
    max_tokens: int = 512,
    rng: random.Random | None = None,
) -> int:
    rng = rng or random.Random()
    return rng.randint(min_tokens, max_tokens)
```

- [ ] **Step 4: Run to verify pass**

```bash
source .venv/bin/activate
pytest tests/loadgen/test_sampling.py -v
```

Expected: PASS (3 passed).

- [ ] **Step 5: Write the failing arrivals test**

`tests/loadgen/test_arrivals.py`:

```python
import random

import pytest

from loadgen.arrivals import poisson_arrival_times


def test_poisson_arrival_times_within_duration_and_sorted():
    rng = random.Random(7)
    times = poisson_arrival_times(5.0, 60.0, rng=rng)
    assert all(0 <= t <= 60.0 for t in times)
    assert times == sorted(times)


def test_poisson_arrival_times_mean_rate_close_to_target():
    rng = random.Random(7)
    times = poisson_arrival_times(rate_per_sec=10.0, duration_sec=1000.0, rng=rng)
    observed_rate = len(times) / 1000.0
    assert 9.0 <= observed_rate <= 11.0


def test_poisson_arrival_times_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        poisson_arrival_times(0.0, 10.0)
```

- [ ] **Step 6: Run to verify failure**

```bash
source .venv/bin/activate
pytest tests/loadgen/test_arrivals.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'loadgen.arrivals'`.

- [ ] **Step 7: Write `loadgen/arrivals.py`**

```python
from __future__ import annotations

import random


def poisson_arrival_times(
    rate_per_sec: float,
    duration_sec: float,
    rng: random.Random | None = None,
) -> list[float]:
    if rate_per_sec <= 0:
        raise ValueError("rate_per_sec must be positive")
    rng = rng or random.Random()
    times: list[float] = []
    t = 0.0
    while True:
        t += rng.expovariate(rate_per_sec)
        if t > duration_sec:
            break
        times.append(t)
    return times
```

- [ ] **Step 8: Run to verify pass**

```bash
source .venv/bin/activate
pytest tests/loadgen/test_arrivals.py -v
```

Expected: PASS (3 passed).

- [ ] **Step 9: Commit**

```bash
git add loadgen/sampling.py loadgen/arrivals.py tests/loadgen/test_sampling.py tests/loadgen/test_arrivals.py
git commit -m "Add load generator request sampling and Poisson arrival scheduling"
```

---

### Task 6: Load generator HTTP client with latency/TTFT metrics

**Files:**
- Create: `loadgen/client.py`
- Test: `tests/loadgen/test_client.py`

**Interfaces:**
- Consumes: nothing beyond `httpx`.
- Produces: `RequestResult` (fields: `context_tokens: int`, `requested_output_tokens: int`, `completed_output_tokens: int`, `latency_sec: float`, `ttft_sec: float | None`, `success: bool`, `error: str | None`), `async send_completion_request(client: httpx.AsyncClient, base_url: str, served_model_name: str, prompt: str, context_tokens: int, max_tokens: int, timeout_sec: float) -> RequestResult`. Task 7's runner imports both.

- [ ] **Step 1: Write the failing client tests**

`tests/loadgen/test_client.py`:

```python
import httpx
import pytest
import respx

from loadgen.client import send_completion_request

SSE_BODY = (
    b'data: {"choices":[{"text":"Hello"}]}\n\n'
    b'data: {"choices":[{"text":" world"}]}\n\n'
    b'data: [DONE]\n\n'
)


@respx.mock
async def test_send_completion_request_counts_tokens_and_records_ttft():
    respx.post("http://localhost:8000/v1/completions").mock(
        return_value=httpx.Response(
            200, content=SSE_BODY, headers={"content-type": "text/event-stream"}
        )
    )
    async with httpx.AsyncClient() as client:
        result = await send_completion_request(
            client,
            base_url="http://localhost:8000",
            served_model_name="qwen3-4b-instruct-2507",
            prompt="hi",
            context_tokens=10,
            max_tokens=50,
            timeout_sec=5.0,
        )
    assert result.success is True
    assert result.completed_output_tokens == 2
    assert result.ttft_sec is not None
    assert result.latency_sec >= result.ttft_sec
    assert result.error is None


@respx.mock
async def test_send_completion_request_records_failure_on_http_error():
    respx.post("http://localhost:8000/v1/completions").mock(return_value=httpx.Response(500))
    async with httpx.AsyncClient() as client:
        result = await send_completion_request(
            client,
            base_url="http://localhost:8000",
            served_model_name="qwen3-4b-instruct-2507",
            prompt="hi",
            context_tokens=10,
            max_tokens=50,
            timeout_sec=5.0,
        )
    assert result.success is False
    assert result.completed_output_tokens == 0
    assert result.ttft_sec is None
    assert result.error is not None
```

- [ ] **Step 2: Run to verify failure**

```bash
source .venv/bin/activate
pytest tests/loadgen/test_client.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'loadgen.client'`.

- [ ] **Step 3: Write `loadgen/client.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
source .venv/bin/activate
pytest tests/loadgen/test_client.py -v
```

Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add loadgen/client.py tests/loadgen/test_client.py
git commit -m "Add async load generator HTTP client with latency/TTFT metrics"
```

---

### Task 7: Load generator runner, CSV output, CLI, and first real baseline run

**Files:**
- Create: `loadgen/runner.py`
- Create: `loadgen/cli.py`
- Test: `tests/loadgen/test_runner.py`
- Create (generated, not hand-written): `results/raw/baseline.csv`

**Interfaces:**
- Consumes: `configs.loader.load_model_config` (Task 2), `loadgen.arrivals.poisson_arrival_times`, `loadgen.sampling.DEFAULT_BUCKETS`, `loadgen.sampling.sample_context_length`, `loadgen.sampling.sample_output_length` (Task 5), `loadgen.client.RequestResult`, `loadgen.client.send_completion_request` (Task 6).
- Produces: `SweepConfig` (fields: `base_url: str`, `served_model_name: str`, `rate_per_sec: float`, `duration_sec: float`, `eviction_aggressiveness: str`, `slo_latency_sec: float`, `timeout_sec: float = 30.0`), `CSV_FIELDS: list[str]`, `result_to_row(result: RequestResult, cfg: SweepConfig) -> dict`, `write_csv(rows: list[dict], path: Path) -> None`, `async run_sweep(cfg: SweepConfig) -> list[dict]`. This is the last task in this plan — later plans (eviction sweeps) reuse `SweepConfig`, `run_sweep`, and `write_csv` as-is, varying only `eviction_aggressiveness`.

- [ ] **Step 1: Write the failing runner tests (CSV/row logic only — no live server needed)**

`tests/loadgen/test_runner.py`:

```python
import csv
from pathlib import Path

from loadgen.client import RequestResult
from loadgen.runner import CSV_FIELDS, SweepConfig, result_to_row, write_csv

CFG = SweepConfig(
    base_url="http://localhost:8000",
    served_model_name="qwen3-4b-instruct-2507",
    rate_per_sec=2.0,
    duration_sec=10.0,
    eviction_aggressiveness="baseline",
    slo_latency_sec=2.0,
)


def test_result_to_row_marks_slo_met_when_within_latency():
    result = RequestResult(100, 50, 50, 1.5, 0.2, True, None)
    row = result_to_row(result, CFG)
    assert row["met_slo"] is True
    assert row["success"] is True


def test_result_to_row_marks_slo_missed_when_over_latency():
    result = RequestResult(100, 50, 50, 3.0, 0.2, True, None)
    row = result_to_row(result, CFG)
    assert row["met_slo"] is False


def test_result_to_row_failed_request_never_meets_slo():
    result = RequestResult(100, 50, 0, 0.1, None, False, "boom")
    row = result_to_row(result, CFG)
    assert row["met_slo"] is False


def test_write_csv_creates_file_with_header_and_rows(tmp_path: Path):
    rows = [result_to_row(RequestResult(100, 50, 50, 1.0, 0.1, True, None), CFG)]
    out = tmp_path / "nested" / "out.csv"
    write_csv(rows, out)
    with out.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_FIELDS
        read_rows = list(reader)
    assert len(read_rows) == 1
    assert read_rows[0]["eviction_aggressiveness"] == "baseline"
```

- [ ] **Step 2: Run to verify failure**

```bash
source .venv/bin/activate
pytest tests/loadgen/test_runner.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'loadgen.runner'`.

- [ ] **Step 3: Write `loadgen/runner.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
source .venv/bin/activate
pytest tests/loadgen/test_runner.py -v
```

Expected: PASS (4 passed).

- [ ] **Step 5: Write the CLI entrypoint**

`loadgen/cli.py`:

```python
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
```

- [ ] **Step 6: Run a real baseline sweep against the live server from Task 4**

With the server from Task 4 still running (relaunch it if not: `python serving/server.py --model-config configs/models/qwen3-4b-instruct-2507-awq.yaml --hardware-config configs/hardware/rtx5070-laptop.yaml`), in another terminal:

```bash
source .venv/bin/activate
python -m loadgen.cli \
  --model-config configs/models/qwen3-4b-instruct-2507-awq.yaml \
  --rate 1.0 \
  --duration 30 \
  --eviction-aggressiveness baseline \
  --slo-latency-sec 5.0 \
  --out results/raw/baseline.csv
```

Expected: prints `wrote N rows to results/raw/baseline.csv` with N roughly `30 * 1.0 = 30` (Poisson variance means it won't be exact). Open the CSV and sanity-check: `success` is `True` for the large majority of rows, `latency_sec` values are positive and plausible (not all identical, not absurdly large), `context_tokens` values span the configured short/medium/long buckets.

- [ ] **Step 7: Commit**

```bash
git add loadgen/runner.py loadgen/cli.py tests/loadgen/test_runner.py results/raw/baseline.csv
git commit -m "Add load generator runner, CSV output, CLI, and first baseline sweep"
```

---

## After this plan

Milestones 1-3 are done: a config-driven vLLM server is running Qwen3-4B-Instruct-2507 (AWQ) on the RTX 5070, and `results/raw/baseline.csv` holds real, sanity-checked throughput/latency numbers with no eviction policy active yet. The next plan covers design-doc milestone 4 onward (StreamingLLM-style eviction policy implementation and integration into vLLM's `KVCacheManager`) — its first task will need to inspect the actual installed vLLM version's source (pinned in Task 3, Step 5 above) to find the real eviction hook point, which is why it wasn't written until that version was known.
