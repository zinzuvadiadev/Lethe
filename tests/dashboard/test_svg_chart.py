from dashboard.svg_chart import latency_distribution_svg


def test_latency_distribution_svg_has_one_rect_per_bin():
    svg = latency_distribution_svg([1.0, 2.0, 3.0, 4.0, 5.0], n_bins=5)
    assert svg.count("<rect") == 5


def test_latency_distribution_svg_empty_input_returns_valid_empty_svg():
    svg = latency_distribution_svg([])
    assert svg.startswith("<svg")
    assert svg.count("<rect") == 0


def test_latency_distribution_svg_all_identical_values_does_not_crash():
    svg = latency_distribution_svg([2.0, 2.0, 2.0], n_bins=4)
    assert svg.count("<rect") == 4


def test_latency_distribution_svg_bars_are_rounded():
    svg = latency_distribution_svg([1.0, 2.0, 3.0], n_bins=3)
    assert 'rx="1.5"' in svg


def test_latency_distribution_svg_includes_baseline_axis_even_when_empty():
    svg = latency_distribution_svg([])
    assert 'class="chart-axis"' in svg
