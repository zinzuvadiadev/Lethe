from __future__ import annotations


def latency_distribution_svg(
    latencies: list[float],
    width: int = 320,
    height: int = 80,
    n_bins: int = 12,
) -> str:
    baseline = f'<line x1="0" y1="{height - 0.5}" x2="{width}" y2="{height - 0.5}" class="chart-axis" />'
    if not latencies:
        return (
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f"{baseline}</svg>"
        )

    lo, hi = min(latencies), max(latencies)
    span = (hi - lo) or 1.0
    bin_width = span / n_bins
    counts = [0] * n_bins
    for v in latencies:
        idx = min(int((v - lo) / bin_width), n_bins - 1)
        counts[idx] += 1
    max_count = max(counts) or 1

    bar_w = width / n_bins
    bars = [baseline]
    for i, c in enumerate(counts):
        bar_h = (c / max_count) * (height - 6)
        x = i * bar_w
        y = height - bar_h - 1
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_w - 1, 0):.1f}" '
            f'height="{bar_h:.1f}" rx="1.5" />'
        )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(bars)
        + "</svg>"
    )
