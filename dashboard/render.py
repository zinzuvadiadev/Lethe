from __future__ import annotations

import html as html_lib

from dashboard.milestones import Milestone
from dashboard.narrative import CommitEntry
from dashboard.svg_chart import latency_distribution_svg
from dashboard.trials import TrialReport

_STYLE = """
:root {
  --bg: #eef1f4; --paper: #ffffff; --paper-inset: #f6f8fa;
  --ink: #1a222b; --ink-muted: #55626f; --ink-faint: #8993a1;
  --line: #dde3e8; --line-strong: #c3cbd2;
  --brand: #2c5f8a; --brand-soft: #dfe9f2;
  --accent: #a8701f; --accent-soft: #f3e6d2;
  --success: #2f7d52; --success-soft: #e0f0e6;
  --warning: #a8701f; --warning-soft: #f3e6d2;
  --danger: #b3432f; --danger-soft: #f7e3de;
  --code-bg: #eef2f5;
  --shadow: 0 1px 2px rgba(20,30,40,.05), 0 4px 14px rgba(20,30,40,.05);
  --serif: Georgia, "Noto Serif", serif;
  --sans: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #10151a; --paper: #171d24; --paper-inset: #12171d;
    --ink: #e8ecef; --ink-muted: #9aa7b3; --ink-faint: #6d7986;
    --line: #262e36; --line-strong: #37424c;
    --brand: #6ea4d1; --brand-soft: #1c2b38;
    --accent: #d99a3f; --accent-soft: #2e2515;
    --success: #5fae83; --success-soft: #16281f;
    --warning: #d99a3f; --warning-soft: #2e2515;
    --danger: #d97c62; --danger-soft: #301c17;
    --code-bg: #0f1418;
    --shadow: 0 1px 2px rgba(0,0,0,.35), 0 8px 24px rgba(0,0,0,.4);
  }
}
* { box-sizing: border-box; }
html { scroll-padding-top: 1.5rem; }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
body {
  background: var(--bg); color: var(--ink); font-family: var(--sans);
  margin: 0; padding: 2.5rem 1rem 5rem; line-height: 1.5;
}
.page { max-width: 60rem; margin: 0 auto; }

.masthead { padding: 0 0.25rem 1.5rem; }
.kicker {
  font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--brand); margin: 0 0 0.5rem;
}
h1 {
  font-family: var(--serif); font-weight: 600; font-size: clamp(1.9rem, 4vw, 2.4rem);
  margin: 0 0 0.35rem; letter-spacing: -0.01em; text-wrap: balance;
}
.tagline { color: var(--ink-muted); margin: 0; max-width: 34rem; }
.generated-at {
  font-family: var(--mono); font-size: 0.72rem; color: var(--ink-faint); margin: 0.6rem 0 0;
}

.quicknav {
  display: flex; gap: 1.25rem; flex-wrap: wrap; margin: 1.1rem 0 0;
  font-family: var(--mono); font-size: 0.78rem;
}
.quicknav a { color: var(--ink-muted); text-decoration: none; border-bottom: 1px solid transparent; }
.quicknav a:hover, .quicknav a:focus-visible { color: var(--brand); border-color: var(--brand); }
.quicknav a:focus-visible { outline: 2px solid var(--brand); outline-offset: 3px; border-radius: 2px; }

.progress-card {
  background: var(--paper); border: 1px solid var(--line); border-radius: 10px;
  box-shadow: var(--shadow); padding: 1.1rem 1.3rem; margin: 1.5rem 0 2.2rem;
}
.progress-head {
  display: flex; justify-content: space-between; align-items: baseline;
  font-family: var(--mono); font-size: 0.85rem; margin-bottom: 0.6rem;
}
.progress-head .progress-label { color: var(--ink); font-weight: 600; }
.progress-head .progress-pct { color: var(--ink-faint); }
.progress-track {
  height: 0.5rem; border-radius: 999px; background: var(--paper-inset);
  border: 1px solid var(--line); overflow: hidden;
}
.progress-fill { height: 100%; background: var(--success); border-radius: inherit; }

section { margin: 2.4rem 0; }
h2 {
  font-family: var(--serif); font-weight: 600; font-size: 1.25rem;
  border-bottom: 1px solid var(--line); padding-bottom: 0.5rem; margin: 0 0 1.1rem;
  scroll-margin-top: 1.25rem;
}

/* Milestones */
.milestone-grid {
  list-style: none; padding: 0; margin: 0; display: grid;
  grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: 0.55rem 1.5rem;
}
.milestone {
  display: flex; align-items: center; gap: 0.65rem; font-size: 0.92rem;
}
.m-icon {
  flex: none; width: 1.5rem; height: 1.5rem; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 0.72rem; font-weight: 600;
}
.milestone-done .m-icon { background: var(--success-soft); color: var(--success); }
.milestone-pending .m-icon {
  color: var(--ink-faint); border: 1px solid var(--line-strong); background: transparent;
}
.milestone-done .m-name { color: var(--ink); }
.milestone-pending .m-name { color: var(--ink-faint); }

/* Trials */
.trial-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(19rem, 1fr)); gap: 1rem;
}
.trial-card {
  background: var(--paper); border: 1px solid var(--line); border-radius: 10px;
  box-shadow: var(--shadow); padding: 1.1rem 1.25rem;
}
.trial-head {
  display: flex; justify-content: space-between; align-items: center;
  gap: 0.75rem; margin-bottom: 0.7rem;
}
.trial-head h3 {
  margin: 0; font-family: var(--mono); font-size: 0.92rem; font-weight: 600;
  min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.pill {
  flex: none; font-family: var(--mono); font-size: 0.72rem; font-weight: 600;
  padding: 0.15rem 0.55rem; border-radius: 999px; letter-spacing: 0.01em;
}
.pill-success { background: var(--success-soft); color: var(--success); }
.pill-warning { background: var(--warning-soft); color: var(--warning); }
.pill-danger { background: var(--danger-soft); color: var(--danger); }
.stat-row {
  display: flex; gap: 1.1rem 1.4rem; flex-wrap: wrap; font-family: var(--mono);
  font-size: 0.82rem; color: var(--ink-muted); margin: 0 0 0.7rem;
  font-variant-numeric: tabular-nums;
}
.stat-row b { color: var(--ink); font-weight: 600; }
.chart-wrap { margin-top: 0.4rem; }
.chart-wrap svg { display: block; width: 100%; height: auto; }
.chart-wrap svg rect { fill: var(--brand); }
.chart-wrap svg .chart-axis { stroke: var(--line); stroke-width: 1; }
.chart-caption {
  font-family: var(--mono); font-size: 0.72rem; color: var(--ink-faint); margin: 0.4rem 0 0;
}
table {
  border-collapse: collapse; width: 100%; font-size: 0.83rem; margin-top: 0.4rem;
  font-variant-numeric: tabular-nums;
}
th, td { text-align: left; padding: 0.32rem 0.55rem; border-bottom: 1px solid var(--line); font-family: var(--mono); }
th { color: var(--ink-faint); font-weight: 500; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em; }
tbody tr:nth-child(even) { background: var(--paper-inset); }
.row-count { font-family: var(--mono); font-size: 0.82rem; color: var(--ink-muted); margin: 0 0 0.6rem; }

/* Narrative timeline */
.timeline { position: relative; padding-left: 1.15rem; }
.timeline::before {
  content: ""; position: absolute; left: 0.26rem; top: 0.4rem; bottom: 0.4rem;
  width: 1px; background: var(--line);
}
.commit-entry { position: relative; padding: 0 0 1.2rem 1.05rem; }
.commit-entry:last-child { padding-bottom: 0; }
.commit-entry::before {
  content: ""; position: absolute; left: -1.06rem; top: 0.4rem;
  width: 0.5rem; height: 0.5rem; border-radius: 50%; background: var(--brand);
  box-shadow: 0 0 0 3px var(--bg);
}
.commit-head {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 0 0.6rem;
}
.commit-subject { font-weight: 600; font-size: 0.92rem; }
.commit-meta { font-family: var(--mono); font-size: 0.72rem; color: var(--ink-faint); }
.commit-body {
  color: var(--ink-muted); font-size: 0.85rem; white-space: pre-wrap;
  margin: 0.3rem 0 0;
}

.empty-note { color: var(--ink-faint); font-size: 0.9rem; }
"""


def _success_pill_class(success_rate: float) -> str:
    if success_rate >= 0.99:
        return "pill-success"
    if success_rate >= 0.90:
        return "pill-warning"
    return "pill-danger"


def _render_trial(trial: TrialReport) -> str:
    name = html_lib.escape(trial.filename)
    if trial.is_loadgen_schema and trial.loadgen_summary is not None:
        s = trial.loadgen_summary
        chart = latency_distribution_svg([s.p50_latency_sec, s.p95_latency_sec] * (s.row_count // 2 or 1))
        pill_class = _success_pill_class(s.success_rate)
        return f"""
        <div class="trial-card">
          <div class="trial-head">
            <h3 title="{name}">{name}</h3>
            <span class="pill {pill_class}">{s.success_rate:.0%} success</span>
          </div>
          <div class="stat-row">
            <span><b>{s.row_count}</b> requests</span>
            <span>p50 <b>{s.p50_latency_sec:.2f}s</b></span>
            <span>p95 <b>{s.p95_latency_sec:.2f}s</b></span>
            <span>mean TTFT <b>{s.mean_ttft_sec:.2f}s</b></span>
          </div>
          <div class="chart-wrap">{chart}</div>
          <p class="chart-caption">latency distribution (p50 &rarr; p95)</p>
        </div>
        """
    rows_html = ""
    if trial.preview_rows:
        cols = list(trial.preview_rows[0].keys())
        header = "".join(f"<th>{html_lib.escape(str(c))}</th>" for c in cols)
        body_rows = "".join(
            "<tr>" + "".join(f"<td>{html_lib.escape(str(row.get(c, '')))}</td>" for c in cols) + "</tr>"
            for row in trial.preview_rows
        )
        rows_html = f"<table><thead><tr>{header}</tr></thead><tbody>{body_rows}</tbody></table>"
    return f"""
    <div class="trial-card">
      <div class="trial-head"><h3 title="{name}">{name}</h3></div>
      <p class="row-count"><b>{trial.row_count}</b> rows</p>
      {rows_html}
    </div>
    """


def _render_milestone(milestone: Milestone, done: bool) -> str:
    css_class = "milestone-done" if done else "milestone-pending"
    icon = "&#10003;" if done else str(milestone.number)
    return (
        f'<li class="milestone {css_class}">'
        f'<span class="m-icon">{icon}</span>'
        f'<span class="m-name">{html_lib.escape(milestone.name)}</span>'
        f"</li>"
    )


def _render_commit(commit: CommitEntry) -> str:
    body_html = (
        f'<div class="commit-body">{html_lib.escape(commit.body)}</div>' if commit.body else ""
    )
    return f"""
    <div class="commit-entry">
      <div class="commit-head">
        <span class="commit-subject">{html_lib.escape(commit.subject)}</span>
        <span class="commit-meta">{commit.sha} &middot; {commit.date}</span>
      </div>
      {body_html}
    </div>
    """


def render_dashboard(
    trials: list[TrialReport],
    commits: list[CommitEntry],
    milestones: list[tuple[Milestone, bool]],
    generated_at: str | None = None,
) -> str:
    trials_html = "".join(_render_trial(t) for t in trials) or '<p class="empty-note">No trials yet.</p>'
    milestones_html = "".join(_render_milestone(m, done) for m, done in milestones)
    commits_html = "".join(_render_commit(c) for c in commits) or '<p class="empty-note">No commits yet.</p>'

    done_count = sum(1 for _, done in milestones if done)
    total_count = len(milestones)
    pct = round(100 * done_count / total_count) if total_count else 0
    progress_html = ""
    if total_count:
        progress_html = f"""
        <section class="progress-card">
          <div class="progress-head">
            <span class="progress-label">{done_count} of {total_count} milestones complete</span>
            <span class="progress-pct">{pct}%</span>
          </div>
          <div class="progress-track"><div class="progress-fill" style="width:{pct}%"></div></div>
        </section>
        """

    generated_html = ""
    if generated_at:
        generated_html = f'<p class="generated-at">Regenerated {html_lib.escape(generated_at)}</p>'

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Lethe — Results Dashboard</title>
<style>{_STYLE}</style></head>
<body><div class="page">

<header class="masthead">
  <div class="kicker">Results Dashboard</div>
  <h1>Lethe</h1>
  <p class="tagline">KV-cache eviction policy + throughput/quality benchmark</p>
  {generated_html}
  <nav class="quicknav">
    <a href="#milestones">Milestones</a>
    <a href="#trials">Trials</a>
    <a href="#narrative">Narrative</a>
  </nav>
</header>

{progress_html}

<section id="milestones">
  <h2>Milestones</h2>
  <ol class="milestone-grid">{milestones_html}</ol>
</section>

<section id="trials">
  <h2>Trials</h2>
  <div class="trial-grid">{trials_html}</div>
</section>

<section id="narrative">
  <h2>Narrative</h2>
  <div class="timeline">{commits_html}</div>
</section>

</div></body></html>
"""
