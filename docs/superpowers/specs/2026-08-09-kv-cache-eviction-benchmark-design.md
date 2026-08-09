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

## 3. Model & hardware modularity

The goal is as much benchmark data as possible, not one model on one machine.
Today's environment is the RTX 5070 (8GB) laptop; a second machine with an RTX
4090 (24GB) becomes available once the tool is working end-to-end there. Rather
than hardcode a model name/quantization/GPU assumption into `/serving/`,
`/loadgen/`, and `/eval/`, both are pulled from small config files, and every
result row is tagged with which (model, hardware) combination produced it.

```
/configs/models/qwen3-4b-instruct-2507-awq.yaml
    hf_repo: Qwen/Qwen3-4B-Instruct-2507   (AWQ variant)
    quantization: awq
    native_context_length: 262144

/configs/hardware/rtx5070-laptop.yaml
    gpu_name: RTX 5070 Laptop
    vram_gb: 8
```

Adding a model or a machine later — e.g. `mistral-7b-instruct-awq.yaml` +
`rtx4090.yaml` once that box is available — means adding a config file, not
touching serving/loadgen/eval code. `/results/` keys every row by
`(model, hardware, eviction_aggressiveness)`, so the same plotting code that
produces the primary throughput-vs-quality curve can also facet by model or
hardware once more than one of each exists.

```mermaid
flowchart LR
    MC["/configs/models/*.yaml"] --> RUN["serving / loadgen / eval\n(read config, no hardcoded model)"]
    HC["/configs/hardware/*.yaml"] --> RUN
    RUN --> TAG["results tagged by\n(model, hardware, aggressiveness)"]
    TAG --> NOW["Now: 1 model x 1 GPU\n(Qwen3-4B x RTX 5070)"]
    TAG -.later, same code.-> NEXT["Later: N models x 2 GPUs\n(+ Mistral/Llama-class x RTX 4090)"]
```

**Scope for the implementation plan below:** build and validate the config-driven
tool against today's single (model, hardware) pair. Extending the model/hardware
config set to the 4090 is a distinct follow-on phase (§10, milestone 9) — not
blocked on any redesign, since the config abstraction is what milestones 1-8
are built against from the start.

## 4. System architecture

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

## 5. Eviction policy: StreamingLLM-style

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

## 6. vLLM integration architecture (confirmed against vLLM 0.26.0 source)

§3's original framing — "subclass/wrap `KVCacheManager`" — undersold this once we
actually read the installed source. Three rounds of investigation (see commit
history for the full findings) landed on a concrete, buildable plan that reuses
existing vLLM machinery rather than inventing new attention-kernel masking:

**The key discovery:** vLLM already ships "R-SWA" (Reference Sliding Window
Attention) — `RSWAAttention`/`RSWASpec`/`RSWAManager` — which does *exactly*
"protect a front region + a recent window, evict the middle of a still-running
request's KV cache," including all the attention-kernel-level masking
(`rswa_mask_mod`) needed for correctness. It's just hardcoded today to protect
the *whole prompt* as the front region, for exactly one model (DeepSeek-V2
variants). Our eviction policy is "the same mechanism, with a configurable
sink length instead of always-the-full-prompt" — reuse, not reimplementation.

```mermaid
flowchart TB
    subgraph OURS["Our code (serving/eviction/)"]
        M1["SinkQwen3Attention\n(Qwen3Attention, with attn_cls swapped\nto RSWAAttention when sink_window set)"]
        M2["SinkQwen3ForCausalLM\n(wires the above into Qwen3Model\nvia decoder_layer_type)"]
        M3["Scheduler subclass\n(monkeypatches one bound method:\ncoordinator.remove_skipped_blocks,\nclamps prompt_len → min(sink_len, prompt_len))"]
    end
    subgraph VLLM["Stock vLLM 0.26.0 (untouched installed source)"]
        V1["RSWAAttention / RSWASpec\n(already exists, used verbatim)"]
        V2["RSWAManager.remove_skipped_blocks\n(already exists, used verbatim)"]
        V3["rswa_mask_mod (FlashAttention backend)\n(already exists, used verbatim —\nwe only use FLASH_ATTN in this deployment)"]
    end
    CLI["vllm serve --model-class-overrides ...\n--scheduler-cls ...\n(plain CLI flags, resolved by\nlazy import inside the vllm subprocess)"]

    CLI -->|"model_class_overrides"| M2
    CLI -->|"scheduler_cls"| M3
    M1 --> V1
    M2 --> M1
    M3 -.wraps, doesn't replace.-> V2
    V1 --> V3
```

- **Attention layer** (`SinkQwen3Attention`): a from-scratch `nn.Module`
  reproducing `Qwen3Attention.__init__` with one line changed — construct
  `RSWAAttention(..., rswa_window=W)` instead of `Attention(...)` when a
  window is configured. This mirrors the exact pattern vLLM's own
  `deepseek_v2.py` already uses; not a novel technique.
- **Model class** (`SinkQwen3ForCausalLM`): thin subclass of `Qwen3Model`
  that passes `decoder_layer_type=SinkQwen3DecoderLayer` (itself a thin
  subclass of `Qwen3DecoderLayer` constructing `SinkQwen3Attention`) —
  `Qwen2Model.__init__` (which `Qwen3Model` reuses) already accepts this as
  a constructor parameter, no upstream override needed.
- **Sink boundary**: `RSWAManager.remove_skipped_blocks` treats whatever
  value is passed as `num_prompt_tokens` as the protected-front-region
  boundary — it has no semantic dependency on that value actually being the
  prompt length. There is exactly one place this value is sourced
  (`KVCacheCoordinator.remove_skipped_blocks`, called from
  `KVCacheManager`'s per-step allocation path), so a **single monkeypatch**
  — replacing the already-constructed coordinator instance's
  `remove_skipped_blocks` bound method with a thin wrapper that clamps
  `num_prompt_tokens` to `min(sink_len, num_prompt_tokens)` before calling
  the original — covers every call path (the main per-step allocation path
  *and* the separate connector-cleanup path) with one attribute
  reassignment. Done inside our own `Scheduler` subclass's `__init__`,
  right after `super().__init__()` constructs the real `KVCacheManager`.
- **Wiring into the running server**: both `--model-class-overrides` and
  `--scheduler-cls` are real `vllm serve` CLI flags that accept
  `"module:ClassName"` strings, lazily imported *inside the vllm
  subprocess* when the engine resolves them — no explicit registration call
  needed in our own process, and no vLLM source file is edited. This does
  mean `serving/server.py`'s subprocess launch needs `PYTHONPATH` set to
  include the repo root so `serving.eviction.*` is importable from inside
  that subprocess (same category of fix as the `PATH`/`CUDA_HOME` issues
  already hit in milestone 2).
- **Aggressiveness knob → config**: recent-window size `W` is threaded via
  `hf_config.rswa_window` (same mechanism vLLM's own `UnlimitedOCR` support
  uses — a plain `getattr`-duck-typed attribute, no class check); sink
  length is threaded the same way via our own `hf_config.sink_len`
  attribute, read by the `Scheduler` subclass. Both get set on the model's
  HF config object before engine construction.
- **Scope-limiting fact used throughout**: this deployment only ever
  selects the `FLASH_ATTN` attention backend (confirmed from this
  project's own server logs), so only that backend's `rswa_mask_mod` needs
  to work — no need to touch or test FlexAttention/Triton's copies.

**What this deliberately does NOT do:** free blocks belonging to a request
that's still in its prefill/prompt phase (RSWA's gap logic only ever frees
blocks *behind* the current decode window, never inside an unprocessed
prompt), and does not attempt to keep FlexAttention/Triton backends working
(out of scope — this deployment doesn't use them).

## 7. Request flow (single request, sequence diagram)

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

## 8. Load generator design (`/loadgen/`)

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

## 9. Quality evaluation (`/eval/`)

- **Benchmark:** LongBench, subset selection justified by (a) fitting within our
  configured max sequence length given the 8GB budget, and (b) spanning task types
  — single-doc QA, multi-doc QA, summarization — rather than one category.
- **Metrics:** perplexity **and** LongBench's native task-level scoring (F1/ROUGE/
  accuracy depending on task), both reported — perplexity alone can look fine while
  task accuracy degrades, especially for eviction policies that drop early-context
  detail.
- Run once per eviction aggressiveness setting (including baseline), diffed against
  baseline to produce "% accuracy drop."

## 10. Results (`/results/`)

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

## 11. Repository structure

```
/serving/        - vLLM integration + eviction policy plugin
/loadgen/        - concurrent request load generator
/eval/           - LongBench harness + accuracy scoring
/results/        - raw benchmark output (CSV/JSON) + plots
/configs/models/ - one YAML per model (hf repo, quantization, context length)
/configs/hardware/ - one YAML per machine (GPU name, VRAM budget)
/docs/           - specs and design docs (this file lives here)
/README.md       - problem statement, method, reproducible instructions, results
/.gitignore      - excludes model weights, checkpoints, logs, venvs
```

## 12. Milestone plan (each an independent, committable step)

```mermaid
flowchart TB
    M1["1. Scaffold\ndirs, .gitignore, README skeleton,\n/configs schema"]
    M2["2. Baseline vLLM serving\nconfig-driven: RTX 5070 profile +\nQwen3-4B-AWQ config, smoke test"]
    M3["3. Load generator vs. baseline\nPoisson load, CSV output,\nsanity-checked throughput"]
    M4["4. Eviction policy implemented\nsink+window, toggleable at startup"]
    M5["5. Sweep across aggressiveness\nthroughput-side results collected"]
    M6["6. LongBench subset eval harness\nperplexity + accuracy, baseline\n+ each aggressiveness setting"]
    M7["7. Combine into tradeoff plot\n+ results table"]
    M8["8. README\nmethod, repro commands, results,\nlimitations"]
    M9["9. Phase 2 (later): add RTX 4090\n+ additional model config(s),\nre-run sweep, facet results by\nmodel/hardware"]

    M1 --> M2 --> M3 --> M4 --> M5 --> M6 --> M7 --> M8 -.follow-on, not blocking.-> M9
```

Each milestone is a real commit — no dumping everything as one commit. Milestones
1-8 are this implementation plan's scope: build and validate the config-driven
tool on the hardware in hand (RTX 5070, Qwen3-4B). Milestone 9 — bringing in the
RTX 4090 and additional model configs for more data points — starts once 1-8 are
done and the tool is proven to work; it reuses the same code, adding config files
rather than redesigning anything (§3).

## 13. Known risks / open technical items (not decisions — spikes for milestone 2)

- vLLM's own CUDA kernels (FlashInfer, quantization kernels) for sm_120
  (Blackwell) have been catching up through 2026; may need a recent vLLM release
  or minor build workarounds. PyTorch itself already supports sm_120 natively in
  this environment (2.12+cu130).
- Exact `KVCacheManager` class/method names to override are pinned once vLLM is
  installed and its source inspected for the version we land on.
- LongBench task subset finalized once max sequence length is confirmed against
  actual available KV-cache budget on this GPU.

## 14. Limitations to carry into the README (known up front)

- Single consumer GPU for the milestone 1-8 results, shared with a desktop
  session — available VRAM is not perfectly stable across runs; this is
  disclosed, not hidden. (Addressed for a second data point in milestone 9.)
- Eviction policy is position-based (StreamingLLM-style), not content-aware —
  expected to break down on tasks requiring precise recall of specific early-
  context details that aren't near a sink or the recent window. This is the
  expected, honest failure mode to report on, not a bug to fix.
- Model/hardware coverage starts at one pair (Qwen3-4B x RTX 5070); breadth
  comes from milestone 9 onward, not from this initial pass.
