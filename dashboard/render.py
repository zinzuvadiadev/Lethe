from __future__ import annotations

import html as html_lib

from dashboard.milestones import Milestone
from dashboard.narrative import CommitEntry
from dashboard.svg_chart import latency_distribution_svg
from dashboard.trials import TrialReport

_STYLE = """
:root {
  --bg: #eef1f4; --paper: #ffffff; --paper-inset: #f4f6f8;
  --ink: #1a222b; --ink-muted: #55626f; --ink-faint: #8993a1;
  --line: #d7dde3; --accent: #a8701f; --accent-soft: #f3e6d2;
  --kept: #2c5f8a; --kept-soft: #dfe9f2; --code-bg: #eef2f5;
  --serif: Georgia, "Noto Serif", serif;
  --sans: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #12161b; --paper: #191f26; --paper-inset: #141a20;
    --ink: #e7ebef; --ink-muted: #9aa7b3; --ink-faint: #6d7986;
    --line: #2b333c; --accent: #d99a3f; --accent-soft: #2e2515;
    --kept: #6ea4d1; --kept-soft: #1c2b38; --code-bg: #10151a;
  }
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); margin: 0; padding: 2.5rem 1rem 5rem; }
.sheet { max-width: 52rem; margin: 0 auto; background: var(--paper); border: 1px solid var(--line); border-radius: 3px; padding: 2.5rem clamp(1.25rem, 5vw, 3rem); }
h1 { font-family: var(--serif); font-size: 1.8rem; margin: 0 0 1.5rem; }
h2 { font-family: var(--serif); font-size: 1.2rem; border-bottom: 1px solid var(--line); padding-bottom: 0.4rem; margin: 2.2rem 0 1rem; }
.card { background: var(--paper-inset); border: 1px solid var(--line); border-radius: 5px; padding: 1rem 1.2rem; margin: 0.8rem 0; }
.card h3 { margin: 0 0 0.5rem; font-family: var(--mono); font-size: 0.95rem; }
.stat-row { display: flex; gap: 1.5rem; flex-wrap: wrap; font-family: var(--mono); font-size: 0.85rem; color: var(--ink-muted); margin: 0.4rem 0; }
.stat-row b { color: var(--ink); }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; margin-top: 0.5rem; }
th, td { text-align: left; padding: 0.3rem 0.6rem; border-bottom: 1px solid var(--line); font-family: var(--mono); }
th { color: var(--ink-faint); font-weight: 500; }
.milestone-list { list-style: none; padding: 0; }
.milestone-list li { display: flex; gap: 0.6rem; align-items: baseline; padding: 0.35rem 0; font-family: var(--mono); font-size: 0.9rem; }
.milestone-done { color: var(--kept); }
.milestone-pending { color: var(--ink-faint); }
.commit { border-top: 1px solid var(--line); padding: 0.7rem 0; }
.commit .subject { font-weight: 600; }
.commit .meta { font-family: var(--mono); font-size: 0.75rem; color: var(--ink-faint); }
.commit .body { color: var(--ink-muted); font-size: 0.88rem; white-space: pre-wrap; margin-top: 0.3rem; }
svg rect { fill: var(--kept); }
"""


def _render_trial(trial: TrialReport) -> str:
    name = html_lib.escape(trial.filename)
    if trial.is_loadgen_schema and trial.loadgen_summary is not None:
        s = trial.loadgen_summary
        chart = latency_distribution_svg([s.p50_latency_sec, s.p95_latency_sec] * (s.row_count // 2 or 1))
        return f"""
        <div class="card">
          <h3>{name}</h3>
          <div class="stat-row">
            <span><b>{s.row_count}</b> requests</span>
            <span><b>{s.success_rate:.0%}</b> success</span>
            <span>p50 <b>{s.p50_latency_sec:.2f}s</b></span>
            <span>p95 <b>{s.p95_latency_sec:.2f}s</b></span>
            <span>mean TTFT <b>{s.mean_ttft_sec:.2f}s</b></span>
          </div>
          {chart}
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
    <div class="card">
      <h3>{name}</h3>
      <div class="stat-row"><span><b>{trial.row_count}</b> rows</span></div>
      {rows_html}
    </div>
    """


def _render_milestone(milestone: Milestone, done: bool) -> str:
    css_class = "milestone-done" if done else "milestone-pending"
    marker = "[x]" if done else "[ ]"
    return f'<li class="{css_class}">{marker} {html_lib.escape(milestone.name)}</li>'


def _render_commit(commit: CommitEntry) -> str:
    body_html = f'<div class="body">{html_lib.escape(commit.body)}</div>' if commit.body else ""
    return f"""
    <div class="commit">
      <div class="subject">{html_lib.escape(commit.subject)}</div>
      <div class="meta">{commit.sha} &middot; {commit.date}</div>
      {body_html}
    </div>
    """


def render_dashboard(
    trials: list[TrialReport],
    commits: list[CommitEntry],
    milestones: list[tuple[Milestone, bool]],
) -> str:
    trials_html = "".join(_render_trial(t) for t in trials) or "<p>No trials yet.</p>"
    milestones_html = "".join(_render_milestone(m, done) for m, done in milestones) or "<p>No milestones tracked.</p>"
    commits_html = "".join(_render_commit(c) for c in commits) or "<p>No commits yet.</p>"

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>KV-Cache Benchmark — Results Dashboard</title>
<style>{_STYLE}</style></head>
<body><div class="sheet">
<h1>KV-Cache Benchmark — Results Dashboard</h1>

<h2>Trials</h2>
{trials_html}

<h2>Milestones</h2>
<ul class="milestone-list">{milestones_html}</ul>

<h2>Narrative</h2>
{commits_html}

</div></body></html>
"""
