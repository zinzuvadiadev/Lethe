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
