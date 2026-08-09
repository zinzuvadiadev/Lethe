# Design: KV-Cache Eviction Policy + Throughput/Quality Benchmark

**Date:** 2026-08-09
**Status:** Draft — pending review

## 1. Problem

Long-context LLM serving is memory-bound: the KV cache grows linearly with context
length and batch size. Eviction/compression methods (StreamingLLM, H2O, etc.) are
usually benchmarked on perplexity alone, in isolation from a real serving stack.
Almost no public work shows the actual **tokens/sec vs. quality-loss curve** under
realistic concurrent load — which is what a practitioner actually needs to decide
whether to adopt a technique.

This project builds one eviction policy, integrates it into a real serving stack
(vLLM), and produces that curve honestly, on real (constrained) consumer hardware.

**Explicit non-goal:** a novel eviction algorithm. Implementing an existing method
correctly and benchmarking it honestly is the contribution.

## 2. Environment constraints that shaped this design

```mermaid
flowchart LR
    A["8GB VRAM\n(RTX 5070 Laptop,\nshared with desktop)"] --> B{Model must be\nsmall + quantized}
    B --> C["Qwen3-4B-Instruct-2507\nAWQ 4-bit (~2.5GB weights)"]
    C --> D["~4-5GB VRAM left\nfor KV cache"]
    D --> E["Enough headroom to actually\nvary context length + concurrency\n= a real tradeoff curve"]

    F["vLLM uses fused attention\nkernels (FlashAttention/FlashInfer)\nfor real throughput"] --> G{Eviction policy must not\nrequire the full attention matrix}
    G --> H["Rules out true H2O\n(needs attention scores)"]
    G --> I["StreamingLLM-style\n(sink tokens + recent window)\nis position-based, no attention\nmatrix needed"]
```

Two decisions fell directly out of hardware/software reality, not preference:

- **Model:** dropped from the original "7-8B" scope to **Qwen3-4B-Instruct-2507
  (AWQ)** — a 7-8B model, even quantized, would leave too little VRAM for the KV
  cache to vary meaningfully, starving the exact axis (context × concurrency) the
  project needs to produce a real curve.
- **Eviction policy:** dropped from the original "H2O" pick to **StreamingLLM-style
  (attention sinks + recent window)** — true H2O requires materializing attention
  scores per token, which is incompatible with the fused attention kernels vLLM
  needs for realistic throughput. vLLM's own (open, unmerged) feature request for a
  pluggable eviction interface — [vllm-project/vllm#36311](https://github.com/vllm-project/vllm/issues/36311)
  — uses attention-sink protection as its reference implementation for exactly this
  reason.

## 3. System architecture

```mermaid
flowchart TB
    subgraph LG["/loadgen/"]
        LGE["Async request generator\nPoisson arrivals, configurable\ncontext-length mix + rate"]
    end

    subgraph SRV["/serving/"]
        API["vLLM OpenAI-compatible\nHTTP server"]
        SCHED["vLLM Scheduler"]
        KVM["KVCacheManager\n(vLLM v1 core, stock)"]
        PLUGIN["EvictionPolicy plugin\n(our code: subclasses/wraps\nKVCacheManager)"]
        API --> SCHED --> KVM
        KVM -. overridden by .-> PLUGIN
    end

    subgraph EVAL["/eval/"]
        LB["LongBench subset harness"]
        SCORE["Perplexity + task accuracy\nscoring, vs. baseline"]
        LB --> SCORE
    end

    subgraph RES["/results/"]
        CSV["Raw CSV/JSON\n(per sweep run)"]
        PLOT["Throughput vs. quality\ndegradation plot + table"]
        CSV --> PLOT
    end

    LGE -->|HTTP requests| API
    LGE -->|latency, TTFT,\ntok/s, SLO pass/fail| CSV
    LB -->|HTTP requests| API
    SCORE -->|accuracy deltas| CSV
```

The eviction plugin lives entirely in our `/serving/` package. It is "pluggable" in
the sense of being our own swappable code, toggled at server startup (not an
official vLLM extension point — none is merged yet). This is documented explicitly
in the README, per the "document clearly anywhere you must patch internals"
requirement.

## 4. Eviction policy: StreamingLLM-style

```mermaid
flowchart LR
    subgraph SEQ["Token sequence in KV cache"]
        direction LR
        S1["sink\ntoken(s)"] --- M["middle tokens\n(evicted under pressure)"] --- R["recent window\n(always kept)"]
    end
    S1 -.protected, never evicted.-> NOTE1[" "]
    R -.protected, always kept.-> NOTE2[" "]
    M -.evicted first when\ncache is full.-> NOTE3[" "]
```

- **Sink tokens:** first N tokens of the sequence, protected from eviction (they
  absorb a disproportionate share of attention mass in pre-norm transformers like
  Qwen).
- **Recent window:** the most recent W tokens, always kept.
- **Middle:** everything else — evicted first when the cache is under pressure.
- **Aggressiveness knob:** window size W. Smaller W = more aggressive eviction =
  more throughput/memory headroom, more expected quality loss. This is the
  independent variable swept across the tradeoff curve.

## 5. Request flow (single request, sequence diagram)

```mermaid
sequenceDiagram
    participant LG as Load Generator
    participant API as vLLM HTTP API
    participant SCHED as Scheduler
    participant KVM as KVCacheManager + Eviction Plugin

    LG->>API: POST /completions (prompt, sampled context length)
    API->>SCHED: enqueue request
    SCHED->>KVM: request KV blocks for prefill
    alt cache has room
        KVM-->>SCHED: blocks allocated
    else cache full
        KVM->>KVM: evict middle-window blocks\n(sinks + recent window protected)
        KVM-->>SCHED: blocks allocated after eviction
    end
    SCHED->>SCHED: run prefill + decode steps
    SCHED-->>API: generated tokens (streamed)
    API-->>LG: response + latency/TTFT recorded
    LG->>LG: append record to results CSV
```

## 6. Load generator design (`/loadgen/`)

- Async Python client (`httpx`/`asyncio`) against vLLM's OpenAI-compatible API.
- **Arrivals:** Poisson process, configurable rate (λ) — drives concurrency.
- **Per-request sampling:** context length drawn from a configurable
  short/medium/long bucket distribution; output length similarly configurable.
- **Sweep dimensions:** arrival rate, context-length mix, eviction aggressiveness
  (including a "disabled" run = baseline).
- **Per-request metrics recorded:** latency, time-to-first-token, tokens/sec,
  success vs. timeout against a fixed latency SLO.
- **Output:** CSV, one row per request, tagged with the sweep parameters that
  produced it.

```mermaid
flowchart TB
    CFG["Sweep config\n(rate x context-mix x aggressiveness)"] --> GEN["Poisson arrival generator"]
    GEN --> REQ1["Request\n(sampled context+output len)"]
    GEN --> REQ2["Request"]
    GEN --> REQ3["Request ..."]
    REQ1 & REQ2 & REQ3 --> POOL["Async request pool\n(concurrent in-flight)"]
    POOL --> SRV["vLLM server"]
    SRV --> METRICS["latency, TTFT, tok/s,\nSLO pass/fail"]
    METRICS --> CSV["results/raw/*.csv"]
```

## 7. Quality evaluation (`/eval/`)

- **Benchmark:** LongBench, subset selection justified by (a) fitting within our
  configured max sequence length given the 8GB budget, and (b) spanning task types
  — single-doc QA, multi-doc QA, summarization — rather than one category.
- **Metrics:** perplexity **and** LongBench's native task-level scoring (F1/ROUGE/
  accuracy depending on task), both reported — perplexity alone can look fine while
  task accuracy degrades, especially for eviction policies that drop early-context
  detail.
- Run once per eviction aggressiveness setting (including baseline), diffed against
  baseline to produce "% accuracy drop."

## 8. Results (`/results/`)

```mermaid
flowchart LR
    A["loadgen CSVs\n(throughput/latency per sweep point)"] --> C["Merge by\naggressiveness setting"]
    B["eval CSVs\n(accuracy/perplexity per setting)"] --> C
    C --> D["results table\n(aggressiveness, tok/s, %accuracy drop)"]
    D --> E["Tradeoff plot:\nx = throughput, y = %accuracy drop\none point per aggressiveness setting"]
```

Raw CSV/JSON per run is kept under `/results/raw/`; the plotting script is
deterministic and re-runnable from those files alone (no need to re-run the
benchmark to regenerate the plot).

## 9. Repository structure

```
/serving/        - vLLM integration + eviction policy plugin
/loadgen/        - concurrent request load generator
/eval/           - LongBench harness + accuracy scoring
/results/        - raw benchmark output (CSV/JSON) + plots
/docs/           - specs and design docs (this file lives here)
/README.md       - problem statement, method, reproducible instructions, results
/.gitignore      - excludes model weights, checkpoints, logs, venvs
```

## 10. Milestone plan (each an independent, committable step)

```mermaid
flowchart TB
    M1["1. Scaffold\ndirs, .gitignore, README skeleton"]
    M2["2. Baseline vLLM serving\ninstall on RTX 5070, verify sm_120,\nserve Qwen3-4B-AWQ, smoke test"]
    M3["3. Load generator vs. baseline\nPoisson load, CSV output,\nsanity-checked throughput"]
    M4["4. Eviction policy implemented\nsink+window, toggleable at startup"]
    M5["5. Sweep across aggressiveness\nthroughput-side results collected"]
    M6["6. LongBench subset eval harness\nperplexity + accuracy, baseline\n+ each aggressiveness setting"]
    M7["7. Combine into tradeoff plot\n+ results table"]
    M8["8. README\nmethod, repro commands, results,\nlimitations"]

    M1 --> M2 --> M3 --> M4 --> M5 --> M6 --> M7 --> M8
```

Each milestone is a real commit — no dumping everything as one commit.

## 11. Known risks / open technical items (not decisions — spikes for milestone 2)

- vLLM's own CUDA kernels (FlashInfer, quantization kernels) for sm_120
  (Blackwell) have been catching up through 2026; may need a recent vLLM release
  or minor build workarounds. PyTorch itself already supports sm_120 natively in
  this environment (2.12+cu130).
- Exact `KVCacheManager` class/method names to override are pinned once vLLM is
  installed and its source inspected for the version we land on.
- LongBench task subset finalized once max sequence length is confirmed against
  actual available KV-cache budget on this GPU.

## 12. Limitations to carry into the README (known up front)

- Single consumer GPU, shared with a desktop session — available VRAM is not
  perfectly stable across runs; this is disclosed, not hidden.
- Eviction policy is position-based (StreamingLLM-style), not content-aware —
  expected to break down on tasks requiring precise recall of specific early-
  context details that aren't near a sink or the recent window. This is the
  expected, honest failure mode to report on, not a bug to fix.
- One model size (4B) — the code is structured so a second size could be added
  later for a scaling comparison, but that is out of scope for this pass.
