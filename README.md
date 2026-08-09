# Lethe

KV-cache eviction policy + throughput/quality benchmark for LLM serving.

## Problem

Long-context LLM serving is memory-bound: the KV cache grows linearly with
context length and batch size. Eviction/compression methods (StreamingLLM,
H2O, etc.) are usually benchmarked on perplexity alone, in isolation from a
real serving stack. Almost no public work shows the actual **tokens/sec vs.
quality-loss curve** under realistic concurrent load — which is what a
practitioner actually needs to decide whether to adopt a technique.

Lethe builds one eviction policy, integrates it into a real serving stack
(vLLM), and produces that curve honestly, on real (constrained) consumer
hardware — not a novel algorithm, but an honest end-to-end measurement of
an existing one.

The full design rationale, including the hardware- and software-driven
pivots along the way, lives in
[`docs/superpowers/specs/2026-08-09-kv-cache-eviction-benchmark-design.md`](docs/superpowers/specs/2026-08-09-kv-cache-eviction-benchmark-design.md).

## Method

- **Eviction policy:** StreamingLLM-style — protect N sink tokens + a
  sliding recent window, evict the middle of a request's KV cache while
  it's still generating. Aggressiveness = window size. Chosen over H2O
  because H2O requires materializing attention scores, which is
  incompatible with the fused attention kernels vLLM needs for realistic
  throughput.
- **vLLM integration:** rather than reimplementing eviction from scratch,
  the policy reuses vLLM 0.26.0's existing R-SWA (Reference Sliding Window
  Attention) machinery — which already implements "protect a front region
  + recent window, evict the middle" including the attention-kernel
  masking needed for correctness — generalized from "protect the whole
  prompt" to "protect N sink tokens." No vLLM source file is edited; it's
  wired in via the real `--model-class-overrides`/`--scheduler-cls`
  `vllm serve` CLI flags. See
  [design doc §6](docs/superpowers/specs/2026-08-09-kv-cache-eviction-benchmark-design.md)
  for the full integration architecture.
- **Load generator:** async Poisson-arrival client against vLLM's
  OpenAI-compatible API, sweeping request rate, context-length mix, and
  eviction aggressiveness, recording per-request latency/TTFT/success to
  CSV.
- **Aggressiveness sweep:** an orchestrator launches a server per
  (sink_len, recent_window) setting, load-tests it, tears it down, and
  repeats — fully automated, one command produces all sweep points.
- **Config-driven:** model and hardware definitions live in
  `configs/models/*.yaml` and `configs/hardware/*.yaml` — no model or GPU
  assumptions are hardcoded into `serving/` or `loadgen/`, so adding a
  second model or machine (see Limitations) means adding a config file,
  not touching code.

## Hardware & Models

- **GPU:** NVIDIA GeForce RTX 5070 Laptop, 8GB VRAM — a live desktop
  session shares this GPU, so available VRAM fluctuates between runs.
- **Model:** [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
  AWQ 4-bit, community quant
  [`cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit`](https://huggingface.co/cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit).
- **Serving:** vLLM 0.26.0. `--quantization` is not forced (vLLM
  auto-detects it from the checkpoint); `gpu_memory_utilization=0.75` and
  `max_model_len=6000` were tuned down from more ambitious starting values
  after hitting real CUDA OOM and KV-cache-budget errors on this GPU —
  see the commit history for the exact numbers hit at each setting.
- **Known environment quirks:**
  - FlashInfer's JIT-compiled top-k/top-p sampling kernel fails to build
    on this CUDA-13.3 + sm_120 (Blackwell) combination;
    `serving/server.py` sets `VLLM_USE_FLASHINFER_SAMPLER=0` to fall back
    to vLLM's PyTorch-native sampler instead.
  - Launching a new vLLM server immediately after stopping the previous
    one can race the CUDA driver's GPU memory release and OOM — the
    aggressiveness-sweep orchestrator waits 5s after each teardown to
    avoid this (found live, see commit history).

## Status

Milestones 1-5 of 9 are done:

1. Scaffold
2. Baseline vLLM serving
3. Load generator vs. baseline
4. **Eviction policy** — implemented and live-verified: an identical
   2500-token request sent to an eviction-enabled vs. baseline server
   showed a flat KV-cache-usage plateau vs. linear growth to 4x that
   plateau — decisive proof the policy actually bounds memory, not just a
   smoke test.
5. **Aggressiveness sweep** — automated 5-point sweep (baseline + 4
   window sizes) run end-to-end against the live GPU, 100% request
   success across all points.

Milestones 6-9 (LongBench quality eval, the throughput-vs-quality tradeoff
plot, this README's final pass, and a second GPU/model pair) are not yet
built — the Results section below reflects that honestly.

![Lethe results dashboard](docs/images/dashboard.png)

Regenerate it any time with:

```bash
.venv/bin/python -m dashboard.cli
```

then open `results/dashboard.html` in a browser (fully offline, no server
needed).

## Reproducing

```bash
# from the repo root
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# confirm the GPU + installed vLLM actually work together
.venv/bin/python serving/verify_gpu.py

# launch the config-driven vLLM server (first run downloads ~2.5GB and
# takes several minutes to compile CUDA graphs; subsequent runs are fast)
.venv/bin/python -m serving.server \
  --model-config configs/models/qwen3-4b-instruct-2507-awq.yaml \
  --hardware-config configs/hardware/rtx5070-laptop.yaml

# ...or with eviction enabled (sink_len=64, a 512-token recent window):
.venv/bin/python -m serving.server \
  --model-config configs/models/qwen3-4b-instruct-2507-awq.yaml \
  --hardware-config configs/hardware/rtx5070-laptop.yaml \
  --sink-len 64 --recent-window 512

# in a second terminal, once the server logs "Application startup complete"
.venv/bin/python serving/smoke_test.py

# run a single load sweep against whichever server is currently running
.venv/bin/python -m loadgen.cli \
  --model-config configs/models/qwen3-4b-instruct-2507-awq.yaml \
  --rate 1.0 --duration 30 \
  --eviction-aggressiveness baseline --slo-latency-sec 5.0 \
  --out results/raw/my_run.csv

# ...or run the full 5-point aggressiveness sweep end-to-end (no server
# needs to be running first — the orchestrator launches/tears down each
# one itself; takes ~15-20 minutes on this hardware)
.venv/bin/python -m loadgen.sweep_cli \
  --model-config configs/models/qwen3-4b-instruct-2507-awq.yaml \
  --hardware-config configs/hardware/rtx5070-laptop.yaml \
  --rate 1.0 --duration 30 --slo-latency-sec 5.0 \
  --out-dir results/raw

# regenerate the results dashboard
.venv/bin/python -m dashboard.cli

# run the automated test suite
.venv/bin/pytest -v
```

## Results

The real 5-point aggressiveness sweep (`results/raw/sweep_*.csv`), sink_len
fixed at 64, rate=1.0 req/s over 30s per setting:

| setting | requests | success | p50 latency | p95 latency | mean TTFT |
|---|---|---|---|---|---|
| baseline (no eviction) | 40 | 100% | 5.02s | 9.55s | 0.11s |
| window_2048 | 20 | 100% | 7.13s | 11.68s | 0.11s |
| window_1024 | 28 | 100% | 8.74s | 14.68s | 0.10s |
| window_512 | 43 | 100% | 12.79s | 18.68s | 0.10s |
| window_256 | 35 | 100% | 8.87s | 14.41s | 0.09s |

Request counts differ per setting (Poisson arrival variance is expected);
latency isn't yet monotonic with window size at this sample size — that's
real, noisy small-sample data, not a cleaned-up trend. **This is the
throughput side only.** The actual core deliverable — throughput plotted
against quality degradation — needs milestone 6 (LongBench eval) to
produce the quality side, and milestone 7 to combine them into the
tradeoff curve. Neither is built yet.

_Filled in at milestone 7 — the tradeoff plot itself gets embedded here
once it exists._

## Limitations

- **Single consumer GPU**, shared with a live desktop session — available
  VRAM is not perfectly stable across runs; this is disclosed, not hidden.
  It's directly why `max_model_len` and `gpu_memory_utilization` ended up
  more conservative than originally planned, and why the sweep
  orchestrator needs a settle delay between server launches (see commit
  history for both).
- **Eviction policy is position-based** (StreamingLLM-style), not
  content-aware — expected to break down on tasks requiring precise recall
  of specific early-context details that aren't near a sink or the recent
  window. That's the expected, honest failure mode to report on once
  milestone 6 (LongBench eval) lands, not a bug to fix.
- **No quality measurement yet.** The eviction policy is verified to bound
  memory correctly (milestone 4) and the throughput data above is real,
  but nothing yet measures how much task accuracy it costs — that's
  milestone 6, and it's the piece that turns this from "an eviction policy
  that works" into "an eviction policy with a known cost."
- **Model/hardware coverage starts at one pair** (Qwen3-4B x RTX 5070
  Laptop). A second machine (RTX 4090, 24GB) and model are planned as
  milestone 9, reusing the same config-driven code — not built yet.
