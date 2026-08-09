import random
from pathlib import Path

from configs.loader import load_model_config
from loadgen.sampling import DEFAULT_BUCKETS, ContextLengthBucket, sample_context_length, sample_output_length

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_OUTPUT_TOKENS = 512


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


def test_default_buckets_fit_within_deployed_model_context_budget():
    # DEFAULT_BUCKETS max_tokens is a prompt length; the server rejects any
    # request whose prompt + output tokens exceeds max_context_length. This
    # guards against silently exceeding that again (it happened once already
    # when buckets were originally sized for a 32768 max_model_len that this
    # 8GB GPU's real KV-cache budget couldn't support — see Task 4 commit).
    model_cfg = load_model_config(REPO_ROOT / "configs/models/qwen3-4b-instruct-2507-awq.yaml")
    for bucket in DEFAULT_BUCKETS:
        assert bucket.max_tokens + DEFAULT_MAX_OUTPUT_TOKENS <= model_cfg.max_context_length
