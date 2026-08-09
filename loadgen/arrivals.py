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
