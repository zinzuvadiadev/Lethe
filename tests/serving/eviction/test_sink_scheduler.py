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
