from __future__ import annotations


def latency_distribution_svg(
    latencies: list[float],
    width: int = 320,
    height: int = 80,
    n_bins: int = 12,
) -> str:
    if not latencies:
        return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"></svg>'

    lo, hi = min(latencies), max(latencies)
    span = (hi - lo) or 1.0
    bin_width = span / n_bins
    counts = [0] * n_bins
    for v in latencies:
        idx = min(int((v - lo) / bin_width), n_bins - 1)
        counts[idx] += 1
    max_count = max(counts) or 1

    bar_w = width / n_bins
    bars = []
    for i, c in enumerate(counts):
        bar_h = (c / max_count) * (height - 4)
        x = i * bar_w
        y = height - bar_h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_w - 1, 0):.1f}" '
            f'height="{bar_h:.1f}" />'
        )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(bars)
        + "</svg>"
    )
