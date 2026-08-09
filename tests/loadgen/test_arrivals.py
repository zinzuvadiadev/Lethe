import random

import pytest

from loadgen.arrivals import poisson_arrival_times


def test_poisson_arrival_times_within_duration_and_sorted():
    rng = random.Random(7)
    times = poisson_arrival_times(5.0, 60.0, rng=rng)
    assert all(0 <= t <= 60.0 for t in times)
    assert times == sorted(times)


def test_poisson_arrival_times_mean_rate_close_to_target():
    rng = random.Random(7)
    times = poisson_arrival_times(rate_per_sec=10.0, duration_sec=1000.0, rng=rng)
    observed_rate = len(times) / 1000.0
    assert 9.0 <= observed_rate <= 11.0


def test_poisson_arrival_times_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        poisson_arrival_times(0.0, 10.0)
