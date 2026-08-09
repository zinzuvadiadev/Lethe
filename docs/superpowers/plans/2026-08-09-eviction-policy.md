# Eviction Policy (Milestone 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A StreamingLLM-style eviction policy (protect N sink tokens + a
sliding recent window, evict the middle of a request's KV cache while it's
still generating) that actually bounds a single request's memory — toggleable
at server startup, no vLLM source file edited.

**Architecture:** Reuse vLLM 0.26.0's existing R-SWA (`RSWAAttention`/
`RSWAManager`/`rswa_mask_mod`) machinery — which already implements "protect a
front region + recent window, evict the middle" including attention-kernel
masking — by generalizing its front-anchor from "the whole prompt" to "the
first `sink_len` tokens." Three new files in `serving/eviction/` (a faithful,
minimally-modified reproduction of vLLM's `Qwen3Attention`/`Qwen3ForCausalLM`
classes plus a `Scheduler` subclass with one targeted monkeypatch), wired into
the existing `vllm serve` subprocess launch via the real, documented
`--model-class-overrides` and `--scheduler-cls` CLI flags — no vLLM source
file is edited. Full technical rationale, including what was investigated and
ruled out, is in
[`docs/superpowers/specs/2026-08-09-kv-cache-eviction-benchmark-design.md`](../specs/2026-08-09-kv-cache-eviction-benchmark-design.md)
§6.

**Tech Stack:** Python 3.10, vLLM 0.26.0 internals (read, not edited),
`pytest`.

## Global Constraints

- Design source of truth: `docs/superpowers/specs/2026-08-09-kv-cache-eviction-benchmark-design.md` §5-6.
- No vLLM source file gets edited — everything lives in `serving/eviction/`,
  wired in via `--model-class-overrides`/`--scheduler-cls` CLI flags
  (real, documented vLLM flags — confirmed present in
  `vllm/engine/arg_utils.py`).
- Aggressiveness knobs (`sink_len`, recent window) are read from environment
  variables (`LETHE_SINK_LEN`, `LETHE_RSWA_WINDOW`) set by
  `serving/server.py` on the subprocess — same pattern already used for
  `VLLM_USE_FLASHINFER_SAMPLER` in the previous plan.
- This deployment only ever selects the `FLASH_ATTN` attention backend
  (confirmed from this project's own server logs) — no FlexAttention/Triton
  compatibility work is in scope.
- Model-side code in this plan is a **faithful reproduction** of vLLM's
  installed `Qwen3Attention`/`Qwen3DecoderLayer`/`Qwen3Model`/
  `Qwen3ForCausalLM` (from `.venv/lib/python3.10/site-packages/vllm/model_executor/models/qwen3.py`,
  read directly, not guessed) with the minimum lines changed to swap in
  `RSWAAttention`. Do not "clean up" or restructure the reproduced parts —
  matching vLLM's real source exactly is what makes this safe; vLLM's
  `Attention.__init__` raises `ValueError: Duplicate layer name` if a second
  attention instance is ever constructed under the same layer prefix, which
  rules out any "construct stock, then swap" shortcut.
- **This is genuinely novel vLLM-internals code, unlike the previous plan's
  application-level Python.** Expect Task 6 (live verification) to require
  real debugging iteration — vLLM logs are the ground truth when something
  doesn't work as designed, the same way milestone 2's server launch took
  five real fix cycles before succeeding. Do not treat a first-attempt
  failure in Task 6 as a plan defect; investigate the actual error and fix
  forward, same as before.
- One task = one commit.

---

### Task 1: Attention-class selection helper + `SinkQwen3Attention`

**Files:**
- Create: `serving/eviction/__init__.py`
- Create: `serving/eviction/sink_attention.py`
- Create: `tests/serving/eviction/__init__.py`
- Test: `tests/serving/eviction/test_sink_attention.py`

**Interfaces:**
- Consumes: nothing new (imports vLLM classes directly).
- Produces: `select_attn_cls(attn_type: str, rswa_window: int | None) -> type` (pure, no vLLM engine context needed), `rswa_window_from_env() -> int | None`, `SinkQwen3Attention` (faithful reproduction of `Qwen3Attention` with `attn_cls` selection generalized). Task 2 imports `SinkQwen3Attention`.

- [ ] **Step 1: Write the failing tests for the pure helpers**

`tests/serving/eviction/test_sink_attention.py`:

```python
from vllm.model_executor.layers.attention.encoder_only_attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.attention.rswa_attention import RSWAAttention
from vllm.v1.attention.backend import AttentionType

from serving.eviction.sink_attention import rswa_window_from_env, select_attn_cls


def test_select_attn_cls_encoder_only_ignores_rswa_window():
    assert select_attn_cls(AttentionType.ENCODER_ONLY, rswa_window=128) is EncoderOnlyAttention


def test_select_attn_cls_decoder_with_window_uses_rswa():
    assert select_attn_cls(AttentionType.DECODER, rswa_window=128) is RSWAAttention


def test_select_attn_cls_decoder_without_window_uses_plain_attention():
    assert select_attn_cls(AttentionType.DECODER, rswa_window=None) is Attention


def test_rswa_window_from_env_unset_returns_none(monkeypatch):
    monkeypatch.delenv("LETHE_RSWA_WINDOW", raising=False)
    assert rswa_window_from_env() is None


def test_rswa_window_from_env_set_returns_int(monkeypatch):
    monkeypatch.setenv("LETHE_RSWA_WINDOW", "256")
    assert rswa_window_from_env() == 256
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/serving/eviction/test_sink_attention.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'serving.eviction'`.

- [ ] **Step 3: Create the package and write `serving/eviction/sink_attention.py`**

```bash
mkdir -p serving/eviction tests/serving/eviction
touch serving/eviction/__init__.py tests/serving/eviction/__init__.py
```

`serving/eviction/sink_attention.py`:

```python
from __future__ import annotations

import os
from typing import Any

from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention.encoder_only_attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.attention.rswa_attention import RSWAAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backend import AttentionType

_ = VllmConfig  # re-exported for callers that need the type


def rswa_window_from_env() -> int | None:
    value = os.environ.get("LETHE_RSWA_WINDOW")
    return int(value) if value else None


def select_attn_cls(attn_type: str, rswa_window: int | None) -> type[nn.Module]:
    """Pure decision: which Attention subclass a layer should construct.

    Generalizes vllm.model_executor.models.qwen3.Qwen3Attention's
    attn_cls selection (plain Attention vs. EncoderOnlyAttention) to also
    select RSWAAttention when a recent-window size is configured — see
    design doc §6 for why RSWAAttention (not a from-scratch masking
    implementation) is the right reuse target.
    """
    if attn_type == AttentionType.ENCODER_ONLY:
        return EncoderOnlyAttention
    if rswa_window is not None:
        return RSWAAttention
    return Attention


class SinkQwen3Attention(nn.Module):
    """Faithful reproduction of vllm.model_executor.models.qwen3.Qwen3Attention
    (.venv/lib/python3.10/site-packages/vllm/model_executor/models/qwen3.py,
    read directly — not guessed), with exactly one change: attn_cls selection
    goes through select_attn_cls() instead of a hardcoded Attention/
    EncoderOnlyAttention branch, so it can construct RSWAAttention when a
    recent-window size is configured. Do not restructure the reproduced
    parts (see Global Constraints — vLLM's Attention.__init__ raises on a
    duplicate layer prefix, so this must construct the right class the
    first time, not construct-then-swap).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
        per_layer_sliding_window: int | None = None,
        rswa_window: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        attn_cls = select_attn_cls(attn_type, rswa_window)
        extra_kwargs: dict[str, Any] = (
            {
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {}
        )
        if attn_cls is RSWAAttention:
            extra_kwargs["rswa_window"] = rswa_window
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=per_layer_sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **extra_kwargs,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/pytest tests/serving/eviction/test_sink_attention.py -v
```

Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add serving/eviction/__init__.py serving/eviction/sink_attention.py tests/serving/eviction/__init__.py tests/serving/eviction/test_sink_attention.py
git commit -m "Add SinkQwen3Attention: generalize vLLM's RSWA front-anchor to configurable sink length"
```

---

### Task 2: `SinkQwen3DecoderLayer`, `SinkQwen3Model`, `SinkQwen3ForCausalLM`

**Files:**
- Create: `serving/eviction/sink_model.py`

**Interfaces:**
- Consumes: `serving.eviction.sink_attention.SinkQwen3Attention`, `rswa_window_from_env` (Task 1).
- Produces: `SinkQwen3ForCausalLM` — the class referenced by `--model-class-overrides` in Task 4.

No new pure logic here (it's wiring/reproduction, not decisions), so no
additional unit tests beyond confirming the module imports cleanly — this
class's correctness is verified live in Task 6, the same way `serving/server.py`'s
`main()` was in the previous plan.

- [ ] **Step 1: Write `serving/eviction/sink_model.py`**

Faithful reproduction of `vllm.model_executor.models.qwen3.Qwen3DecoderLayer`/
`Qwen3Model`/`Qwen3ForCausalLM`
(`.venv/lib/python3.10/site-packages/vllm/model_executor/models/qwen3.py`,
read directly), with `Qwen3Attention` → `SinkQwen3Attention` and
`Qwen3DecoderLayer` → `SinkQwen3DecoderLayer` swapped in, and `rswa_window`
threaded from `rswa_window_from_env()`:

```python
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import (
    LocalArgmaxMixin,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.qwen2 import Qwen2MLP as Qwen3MLP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from serving.eviction.sink_attention import SinkQwen3Attention, rswa_window_from_env


class SinkQwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        per_layer_sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = SinkQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
            per_layer_sliding_window=per_layer_sliding_window,
            rswa_window=rswa_window_from_env(),
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        from vllm.model_executor.layers.layernorm import RMSNorm

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class SinkQwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=SinkQwen3DecoderLayer,
        )


class SinkQwen3ForCausalLM(
    LocalArgmaxMixin, nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    hf_to_vllm_mapper = SinkQwen3Model.hf_to_vllm_mapper
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = SinkQwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
.venv/bin/python -c "from serving.eviction.sink_model import SinkQwen3ForCausalLM; print('OK')"
```

Expected: `OK` — this confirms all imports resolve and the class bodies are
syntactically/referentially valid, without needing a live vLLM engine
context (which the constructor itself requires, hence deferring real
construction to Task 6).

- [ ] **Step 3: Commit**

```bash
git add serving/eviction/sink_model.py
git commit -m "Add SinkQwen3ForCausalLM: wire SinkQwen3Attention into a full Qwen3 model subclass"
```

---

### Task 3: `SinkScheduler` — the KV-cache-eviction-boundary monkeypatch

**Files:**
- Create: `serving/eviction/sink_scheduler.py`
- Test: `tests/serving/eviction/test_sink_scheduler.py`

**Interfaces:**
- Consumes: nothing beyond vLLM's `Scheduler`.
- Produces: `sink_len_from_env() -> int | None`, `clamp_num_prompt_tokens(sink_len: int, num_prompt_tokens: int | None) -> int | None` (pure, tested), `SinkScheduler` (the class referenced by `--scheduler-cls` in Task 4).

- [ ] **Step 1: Write the failing tests for the pure helpers**

`tests/serving/eviction/test_sink_scheduler.py`:

```python
from serving.eviction.sink_scheduler import clamp_num_prompt_tokens, sink_len_from_env


def test_clamp_num_prompt_tokens_clamps_when_above_sink_len():
    assert clamp_num_prompt_tokens(sink_len=64, num_prompt_tokens=500) == 64


def test_clamp_num_prompt_tokens_passes_through_when_below_sink_len():
    assert clamp_num_prompt_tokens(sink_len=64, num_prompt_tokens=10) == 10


def test_clamp_num_prompt_tokens_passes_through_none():
    assert clamp_num_prompt_tokens(sink_len=64, num_prompt_tokens=None) is None


def test_sink_len_from_env_unset_returns_none(monkeypatch):
    monkeypatch.delenv("LETHE_SINK_LEN", raising=False)
    assert sink_len_from_env() is None


def test_sink_len_from_env_set_returns_int(monkeypatch):
    monkeypatch.setenv("LETHE_SINK_LEN", "64")
    assert sink_len_from_env() == 64
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/serving/eviction/test_sink_scheduler.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'serving.eviction.sink_scheduler'`.

- [ ] **Step 3: Write `serving/eviction/sink_scheduler.py`**

```python
from __future__ import annotations

import os

from vllm.v1.core.sched.scheduler import Scheduler


def sink_len_from_env() -> int | None:
    value = os.environ.get("LETHE_SINK_LEN")
    return int(value) if value else None


def clamp_num_prompt_tokens(sink_len: int, num_prompt_tokens: int | None) -> int | None:
    """The core new logic: reinterpret RSWA's "protect the whole prompt"
    boundary as "protect only the first sink_len tokens". vLLM's
    RSWAManager.remove_skipped_blocks treats whatever value flows in as
    num_prompt_tokens as the protected-front-region boundary with no
    semantic dependency on it actually being the prompt length (confirmed
    by reading vllm/v1/core/single_type_kv_cache_manager.py directly) — so
    clamping it here is sufficient, no changes needed inside vLLM itself.
    """
    if num_prompt_tokens is None:
        return None
    return min(sink_len, num_prompt_tokens)


class SinkScheduler(Scheduler):
    """Wraps the stock vLLM Scheduler to apply clamp_num_prompt_tokens()
    to every call that reaches KVCacheCoordinator.remove_skipped_blocks.

    Patches exactly one bound method on the already-constructed
    KVCacheCoordinator instance (vllm/v1/core/kv_cache_coordinator.py) —
    this single seam covers both call paths that lead to gap eviction (the
    main per-step allocation path in KVCacheManager.allocate_slots, and the
    separate P/D-connector-cleanup path), since both ultimately call this
    same coordinator instance's remove_skipped_blocks. See design doc §6
    for why "subclass KVCacheManager instead" was ruled out (eviction
    logic is spread across kv_cache_manager.py, kv_cache_coordinator.py,
    and single_type_kv_cache_manager.py with no single clean override
    point below the Scheduler level).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        sink_len = sink_len_from_env()
        if sink_len is None:
            return
        coordinator = self.kv_cache_manager.coordinator
        original_remove_skipped_blocks = coordinator.remove_skipped_blocks

        def patched_remove_skipped_blocks(
            request_id: str,
            processed_computed_tokens: int,
            num_prompt_tokens: int | None = None,
        ) -> None:
            original_remove_skipped_blocks(
                request_id,
                processed_computed_tokens,
                clamp_num_prompt_tokens(sink_len, num_prompt_tokens),
            )

        coordinator.remove_skipped_blocks = patched_remove_skipped_blocks
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/pytest tests/serving/eviction/test_sink_scheduler.py -v
```

Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add serving/eviction/sink_scheduler.py tests/serving/eviction/test_sink_scheduler.py
git commit -m "Add SinkScheduler: clamp the RSWA protected-prefix boundary to sink_len"
```

---

### Task 4: Wire eviction into `serving/server.py`

**Files:**
- Modify: `serving/server.py`
- Modify: `tests/serving/test_server.py`

**Interfaces:**
- Consumes: nothing new (this task only builds CLI args/env vars — it
  references the `serving.eviction.*` module paths as strings, not as
  imports).
- Produces: `build_serve_args(model_cfg, hw_cfg, port=8000, sink_len=None, recent_window=None) -> list[str]` (extended signature — new params default to `None`, preserving existing call sites), and `main()` now sets `LETHE_SINK_LEN`/`LETHE_RSWA_WINDOW` env vars and adds `--sink-len`/`--recent-window` CLI args.

- [ ] **Step 1: Write the failing tests for the extended `build_serve_args`**

Add to `tests/serving/test_server.py` (existing file — append these, don't
remove the existing tests):

```python
def test_build_serve_args_omits_eviction_flags_when_disabled():
    args = build_serve_args(MODEL, SMALL_HW)
    assert "--model-class-overrides" not in args
    assert "--scheduler-cls" not in args


def test_build_serve_args_includes_eviction_flags_when_enabled():
    args = build_serve_args(MODEL, SMALL_HW, sink_len=64, recent_window=256)
    assert "--model-class-overrides" in args
    override_value = args[args.index("--model-class-overrides") + 1]
    assert "serving.eviction.sink_model:SinkQwen3ForCausalLM" in override_value
    assert "Qwen3ForCausalLM" in override_value
    assert args[args.index("--scheduler-cls") + 1] == "serving.eviction.sink_scheduler:SinkScheduler"
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/serving/test_server.py -v -k eviction
```

Expected: FAIL — `TypeError: build_serve_args() got an unexpected keyword argument 'sink_len'`.

- [ ] **Step 3: Extend `serving/server.py`**

Read the current file first (it already has `build_serve_args`, `find_pip_cuda_home`, and `main()` from the previous plan) — this step modifies it in place:

```python
import json
```

Add this import near the top alongside the existing `argparse`/`os`/`subprocess`/`sys`/`Path` imports.

Change the `build_serve_args` signature and body:

```python
def build_serve_args(
    model_cfg: ModelConfig,
    hw_cfg: HardwareConfig,
    port: int = 8000,
    sink_len: int | None = None,
    recent_window: int | None = None,
) -> list[str]:
    # --quantization is deliberately NOT passed here: vLLM auto-detects it
    # from the checkpoint's own config.json, and forcing a value that doesn't
    # exactly match the checkpoint's internal quant_method label (e.g. this
    # AWQ checkpoint declares itself as "compressed-tensors") makes vLLM
    # refuse to start. model_cfg.quantization stays as documentation/results
    # metadata rather than a CLI constraint.
    #
    # 0.75 (not 0.85) for small-VRAM cards: measured on the RTX 5070 laptop
    # (8GB, shared with a live desktop session), 0.85 leaves too thin a
    # margin between vLLM's KV-cache budget estimate and actual allocation
    # (desktop overhead + fragmentation) — observed a real CUDA OOM with
    # only 91MB free during activation-memory allocation at 0.85.
    gpu_util = 0.75 if hw_cfg.vram_gb <= 12 else 0.9
    args = [
        "vllm",
        "serve",
        model_cfg.hf_repo,
        "--served-model-name", model_cfg.served_model_name,
        "--max-model-len", str(model_cfg.max_context_length),
        "--gpu-memory-utilization", str(gpu_util),
        "--port", str(port),
    ]
    if sink_len is not None and recent_window is not None:
        # See docs/superpowers/specs/2026-08-09-kv-cache-eviction-benchmark-design.md
        # §6: reuses vLLM's existing RSWA machinery via a model-class
        # override + scheduler override, both real vllm serve CLI flags
        # resolved by lazy import inside the subprocess (main() sets
        # PYTHONPATH so serving.eviction.* is importable there).
        overrides = json.dumps({"Qwen3ForCausalLM": "serving.eviction.sink_model:SinkQwen3ForCausalLM"})
        args += ["--model-class-overrides", overrides]
        args += ["--scheduler-cls", "serving.eviction.sink_scheduler:SinkScheduler"]
    return args
```

Change `main()`'s argument parsing and env-var setup:

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

    env = os.environ.copy()
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
        print(f"CUDA_HOME not set; using pip-installed CUDA toolkit at {cuda_home}")

    # FlashInfer's JIT-compiled top-k/top-p sampling kernel fails to build on
    # this GPU/CUDA-toolkit combination: its bundled CCCL headers reject the
    # pip-installed CUDA 13.3 nvcc + sm_120f (Blackwell) target with "CUDA
    # compiler and CUDA toolkit headers are incompatible". This falls back to
    # vLLM's PyTorch-native sampler, which doesn't hit that JIT path.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    if args.sink_len is not None:
        env["LETHE_SINK_LEN"] = str(args.sink_len)
    if args.recent_window is not None:
        env["LETHE_RSWA_WINDOW"] = str(args.recent_window)

    print(f"launching: {' '.join(argv)}")
    subprocess.run(argv, check=True, env=env)
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/pytest tests/serving/test_server.py -v
```

Expected: PASS (all tests, existing + 2 new).

- [ ] **Step 5: Run the full test suite**

```bash
.venv/bin/pytest -v
```

Expected: all tests pass (existing 45 + this plan's ~15 new).

- [ ] **Step 6: Commit**

```bash
git add serving/server.py tests/serving/test_server.py
git commit -m "Wire eviction policy into serving/server.py via CLI flags + env vars"
```

---

### Task 5: Update the dashboard milestone marker

**Files:**
- Modify: `dashboard/milestones.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing new — this is a one-line data update, not new logic.

- [ ] **Step 1: Update the milestone 4 marker path**

In `dashboard/milestones.py`, change:

```python
    Milestone(4, "Eviction policy implemented", "serving/eviction.py"),
```

to:

```python
    Milestone(4, "Eviction policy implemented", "serving/eviction/sink_scheduler.py"),
```

(The marker was a guess made before this plan existed — `serving/eviction/`
turned out to be a package, not a single file, and `sink_scheduler.py` is
the last of its modules to land in Task 3, making it a reasonable
"is milestone 4 actually done" signal.)

- [ ] **Step 2: Regenerate the dashboard and confirm milestone 4 flips to done**

```bash
.venv/bin/python -m dashboard.cli
grep -o 'milestone-[a-z]*">\[.\] Eviction policy implemented' results/dashboard.html
```

Expected: `milestone-done">[x] Eviction policy implemented`.

- [ ] **Step 3: Commit**

```bash
git add dashboard/milestones.py
git commit -m "Update milestone 4 marker to match the eviction/ package layout"
```

---

### Task 6: Live verification — launch with eviction enabled, confirm it actually works

This task has no pre-written code — it's real debugging against the live
GPU, the same way `serving/server.py`'s original launch took five real
iterations in the previous plan. Follow the process, not a script.

- [ ] **Step 1: Launch the server with eviction enabled**

Pick small-but-plausible values relative to this deployment's
`max_context_length: 6000` (see `configs/models/qwen3-4b-instruct-2507-awq.yaml`)
— e.g. `--sink-len 64 --recent-window 512` — a real first aggressiveness
setting, not milestone 5's sweep yet:

```bash
.venv/bin/python -m serving.server \
  --model-config configs/models/qwen3-4b-instruct-2507-awq.yaml \
  --hardware-config configs/hardware/rtx5070-laptop.yaml \
  --sink-len 64 --recent-window 512 \
  > /tmp/vllm-eviction-server.log 2>&1 &
```

Watch the log (`tail -f /tmp/vllm-eviction-server.log` or equivalent) for:
- `launching: ... --model-class-overrides {"Qwen3ForCausalLM": ...} --scheduler-cls ...` — confirms Task 4's wiring produced the expected command line.
- Any `ImportError`/`ModuleNotFoundError` mentioning `serving.eviction` — means `PYTHONPATH` isn't reaching the subprocess correctly; check the env dict built in `main()`.
- Any `TypeError`/`AttributeError` during model construction — means the faithful-reproduction assumption in Task 1/2 missed something; compare against the real installed source at `.venv/lib/python3.10/site-packages/vllm/model_executor/models/qwen3.py` again, don't guess a fix.
- Successful startup ends the same way milestone 2's did: `Application startup complete.`

- [ ] **Step 2: Run the smoke test against it**

```bash
.venv/bin/python serving/smoke_test.py
```

Expected: `OK: server responded with a non-empty completion` — same
smoke test as the baseline server, now running with eviction active.
Coherent output here is the real correctness signal (garbled/repetitive
text would indicate the attention masking generalization broke
correctness, not just performance).

- [ ] **Step 3: Confirm eviction is actually engaging, not silently inert**

Send a request whose prompt + generated tokens will meaningfully exceed
`sink_len + recent_window` (e.g. a short prompt with `max_tokens` set high
enough that decoding alone crosses 512+64=576 tokens), and compare GPU
memory behavior against an equivalent request against the baseline
(no-eviction) server from the previous plan. A working eviction policy
should show KV-cache memory leveling off rather than growing linearly with
generated length; a silently-inert one (e.g. the monkeypatch not actually
being hit) would show the same linear growth as baseline. Use
`nvidia-smi --query-gpu=memory.used --format=csv` sampled during
generation, the same tool already used throughout this project's setup
debugging.

- [ ] **Step 4: Record what was actually found**

Whatever the outcome — update this plan's checklist and, if anything
deviated from the design (e.g. a different env var propagation path was
needed, or a vLLM assumption from the investigation turned out subtly
wrong), add a note to design doc §6 documenting the real behavior, the
same way milestone 2's real CUDA/quantization/sampler findings got folded
back into the design doc rather than left only in commit messages.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Verify eviction policy end-to-end against the live server"
```

(Exact files staged depend on what Step 4 produced — could be just a
design-doc update, or could include fixes to Tasks 1-4's code if the live
run surfaced a real bug.)

---

## After this plan

Milestone 4 (design doc §12) is done: the eviction policy exists, is
toggleable at server startup via `--sink-len`/`--recent-window`, and has
been verified to actually bound memory on the live server. The next plan
covers milestone 5 (sweep across aggressiveness settings) — reusing
`loadgen`'s existing `SweepConfig`/`run_sweep`/`write_csv` unchanged, just
running it once per `(sink_len, recent_window)` pair against a server
launched with that pair's flags, and milestone 6 (LongBench eval harness) to
measure the quality side of the tradeoff this eviction policy exists to
produce.
