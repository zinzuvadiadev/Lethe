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
