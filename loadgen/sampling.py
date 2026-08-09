from __future__ import annotations

import dataclasses
import random


@dataclasses.dataclass(frozen=True)
class ContextLengthBucket:
    name: str
    min_tokens: int
    max_tokens: int
    weight: float


# Bucket top capped well under this deployment's max_model_len (6000, see
# configs/models/qwen3-4b-instruct-2507-awq.yaml) minus room for output
# tokens (up to 512 by default, see sample_output_length) — otherwise the
# load generator would send requests the live server rejects as too long.
DEFAULT_BUCKETS: tuple[ContextLengthBucket, ...] = (
    ContextLengthBucket("short", 128, 512, 0.5),
    ContextLengthBucket("medium", 512, 2048, 0.35),
    ContextLengthBucket("long", 2048, 4000, 0.15),
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
