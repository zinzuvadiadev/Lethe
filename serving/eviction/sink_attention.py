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
