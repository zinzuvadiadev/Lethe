# Results Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `dashboard/` package and CLI that scans `results/raw/*.csv`, the git commit log, and the repo's milestone markers, and renders one self-contained, offline-viewable HTML report (`results/dashboard.html`).

**Architecture:** Four small, independently-testable modules (`trials.py`, `svg_chart.py`, `milestones.py`, `narrative.py`) each produce plain dataclasses/strings with no cross-dependencies; `render.py` combines their outputs into HTML; `cli.py` wires it to a command. No server, no database, no new runtime dependency beyond `pandas` (already required).

**Tech Stack:** Python 3.10, pandas (CSV parsing/stats), stdlib `subprocess` (git log), hand-written inline SVG (no charting library), pytest.

## Global Constraints

- Design source of truth: `docs/superpowers/specs/2026-08-09-results-dashboard-design.md`.
- Local-only, single-user, no accounts, no server — regenerate `results/dashboard.html` on demand by re-running the CLI.
- `results/dashboard.html` is a derived artifact — gitignore it, don't commit it. The `dashboard/` source is the tracked deliverable.
- Reuse `loadgen.runner.CSV_FIELDS` to detect the loadgen-sweep CSV schema rather than duplicating the column list — keeps the two in sync automatically.
- One task = one commit.

---

### Task 1: Trial discovery, schema detection, and summary stats

**Files:**
- Create: `dashboard/__init__.py`
- Create: `dashboard/trials.py`
- Create: `tests/dashboard/__init__.py`
- Test: `tests/dashboard/test_trials.py`

**Interfaces:**
- Consumes: `loadgen.runner.CSV_FIELDS` (existing).
- Produces: `LoadgenSummary` (fields: `row_count: int`, `success_rate: float`, `p50_latency_sec: float`, `p95_latency_sec: float`, `mean_ttft_sec: float`), `TrialReport` (fields: `filename: str`, `is_loadgen_schema: bool`, `row_count: int`, `columns: list[str]`, `loadgen_summary: LoadgenSummary | None`, `preview_rows: list[dict]`), `is_loadgen_schema(columns: set[str]) -> bool`, `load_trial(csv_path: Path) -> TrialReport`, `discover_trials(results_raw_dir: Path) -> list[TrialReport]`. Task 5 imports `discover_trials` and both dataclasses.

- [ ] **Step 1: Write the failing tests**

`tests/dashboard/test_trials.py`:

```python
import csv
from pathlib import Path

from dashboard.trials import TrialReport, discover_trials, is_loadgen_schema, load_trial
from loadgen.runner import CSV_FIELDS


def _write_loadgen_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


LOADGEN_ROWS = [
    {
        "eviction_aggressiveness": "baseline", "rate_per_sec": 1.0,
        "context_tokens": 100, "requested_output_tokens": 50,
        "completed_output_tokens": 50, "latency_sec": v,
        "ttft_sec": 0.1, "success": success, "met_slo": True, "error": "",
    }
    for v, success in [(1.0, True), (2.0, True), (3.0, True), (4.0, False)]
]


def test_is_loadgen_schema_true_for_matching_columns():
    assert is_loadgen_schema(set(CSV_FIELDS)) is True


def test_is_loadgen_schema_false_for_other_columns():
    assert is_loadgen_schema({"task", "accuracy", "perplexity"}) is False


def test_load_trial_computes_loadgen_summary(tmp_path: Path):
    csv_path = tmp_path / "baseline.csv"
    _write_loadgen_csv(csv_path, LOADGEN_ROWS)
    report = load_trial(csv_path)
    assert report.filename == "baseline.csv"
    assert report.is_loadgen_schema is True
    assert report.row_count == 4
    assert report.loadgen_summary is not None
    assert report.loadgen_summary.success_rate == 0.75
    assert report.loadgen_summary.p50_latency_sec == 2.5
    assert abs(report.loadgen_summary.p95_latency_sec - 3.85) < 1e-9


def test_load_trial_falls_back_to_generic_table_for_unknown_schema(tmp_path: Path):
    csv_path = tmp_path / "eval_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "accuracy"])
        writer.writeheader()
        writer.writerow({"task": "qa", "accuracy": 0.82})
    report = load_trial(csv_path)
    assert report.is_loadgen_schema is False
    assert report.loadgen_summary is None
    assert report.row_count == 1
    assert report.preview_rows == [{"task": "qa", "accuracy": 0.82}]


def test_discover_trials_finds_all_csvs_sorted(tmp_path: Path):
    _write_loadgen_csv(tmp_path / "b.csv", LOADGEN_ROWS)
    _write_loadgen_csv(tmp_path / "a.csv", LOADGEN_ROWS)
    reports = discover_trials(tmp_path)
    assert [r.filename for r in reports] == ["a.csv", "b.csv"]


def test_discover_trials_empty_dir_returns_empty_list(tmp_path: Path):
    assert discover_trials(tmp_path) == []
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/dashboard/test_trials.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard'`.

- [ ] **Step 3: Write `dashboard/__init__.py` and `tests/dashboard/__init__.py`**

Both empty files, so `dashboard` and `tests.dashboard` are importable packages.

- [ ] **Step 4: Write `dashboard/trials.py`**

```python
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from loadgen.runner import CSV_FIELDS

LOADGEN_SCHEMA_COLUMNS = frozenset(CSV_FIELDS)


@dataclasses.dataclass(frozen=True)
class LoadgenSummary:
    row_count: int
    success_rate: float
    p50_latency_sec: float
    p95_latency_sec: float
    mean_ttft_sec: float


@dataclasses.dataclass(frozen=True)
class TrialReport:
    filename: str
    is_loadgen_schema: bool
    row_count: int
    columns: list[str]
    loadgen_summary: LoadgenSummary | None
    preview_rows: list[dict]


def is_loadgen_schema(columns: set[str]) -> bool:
    return LOADGEN_SCHEMA_COLUMNS.issubset(columns)


def summarize_loadgen(df: pd.DataFrame) -> LoadgenSummary:
    return LoadgenSummary(
        row_count=len(df),
        success_rate=float(df["success"].mean()),
        p50_latency_sec=float(df["latency_sec"].quantile(0.5)),
        p95_latency_sec=float(df["latency_sec"].quantile(0.95)),
        mean_ttft_sec=float(df["ttft_sec"].mean()),
    )


def load_trial(csv_path: Path) -> TrialReport:
    df = pd.read_csv(csv_path)
    loadgen = is_loadgen_schema(set(df.columns))
    return TrialReport(
        filename=csv_path.name,
        is_loadgen_schema=loadgen,
        row_count=len(df),
        columns=list(df.columns),
        loadgen_summary=summarize_loadgen(df) if loadgen else None,
        preview_rows=df.head(20).to_dict(orient="records"),
    )


def discover_trials(results_raw_dir: Path) -> list[TrialReport]:
    return [load_trial(p) for p in sorted(results_raw_dir.glob("*.csv"))]
```

- [ ] **Step 5: Run to verify pass**

```bash
.venv/bin/pytest tests/dashboard/test_trials.py -v
```

Expected: PASS (6 passed).

- [ ] **Step 6: Commit**

```bash
git add dashboard/__init__.py dashboard/trials.py tests/dashboard/__init__.py tests/dashboard/test_trials.py
git commit -m "Add dashboard trial discovery and loadgen-schema summary stats"
```

---

### Task 2: Inline SVG latency-distribution chart

**Files:**
- Create: `dashboard/svg_chart.py`
- Test: `tests/dashboard/test_svg_chart.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `latency_distribution_svg(latencies: list[float], width: int = 320, height: int = 80, n_bins: int = 12) -> str`. Task 5 imports this.

- [ ] **Step 1: Write the failing tests**

`tests/dashboard/test_svg_chart.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/dashboard/test_svg_chart.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.svg_chart'`.

- [ ] **Step 3: Write `dashboard/svg_chart.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/pytest tests/dashboard/test_svg_chart.py -v
```

Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/svg_chart.py tests/dashboard/test_svg_chart.py
git commit -m "Add inline SVG latency-distribution chart for the dashboard"
```

---

### Task 3: Milestone status detection

**Files:**
- Create: `dashboard/milestones.py`
- Test: `tests/dashboard/test_milestones.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Milestone` (fields: `number: int`, `name: str`, `marker_path: str`), `MILESTONES: tuple[Milestone, ...]`, `milestone_status(repo_root: Path) -> list[tuple[Milestone, bool]]`. Task 5 imports `MILESTONES` (indirectly, via `milestone_status`'s return) and `milestone_status`.

- [ ] **Step 1: Write the failing tests**

`tests/dashboard/test_milestones.py`:

```python
from pathlib import Path

from dashboard.milestones import Milestone, milestone_status


def test_milestone_status_marks_existing_marker_done(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "loader.py").touch()
    statuses = milestone_status(tmp_path)
    by_name = {m.name: done for m, done in statuses}
    assert by_name["Config module"] is True


def test_milestone_status_marks_missing_marker_pending(tmp_path: Path):
    statuses = milestone_status(tmp_path)
    by_name = {m.name: done for m, done in statuses}
    assert by_name["Config module"] is False


def test_milestone_status_preserves_milestone_order(tmp_path: Path):
    statuses = milestone_status(tmp_path)
    numbers = [m.number for m, _done in statuses]
    assert numbers == sorted(numbers)


def test_milestone_is_a_plain_dataclass_with_expected_fields():
    m = Milestone(number=1, name="Test", marker_path="foo.py")
    assert (m.number, m.name, m.marker_path) == (1, "Test", "foo.py")
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/dashboard/test_milestones.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.milestones'`.

- [ ] **Step 3: Write `dashboard/milestones.py`**

```python
from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Milestone:
    number: int
    name: str
    marker_path: str


MILESTONES: tuple[Milestone, ...] = (
    Milestone(1, "Scaffold", "pytest.ini"),
    Milestone(2, "Config module", "configs/loader.py"),
    Milestone(3, "GPU + vLLM verification", "serving/verify_gpu.py"),
    Milestone(4, "Config-driven server + smoke test", "serving/server.py"),
    Milestone(5, "Load generator sampling", "loadgen/sampling.py"),
    Milestone(6, "Load generator HTTP client", "loadgen/client.py"),
    Milestone(7, "Load generator runner + baseline run", "results/raw/baseline.csv"),
    Milestone(8, "LongBench eval harness", "eval/harness.py"),
    Milestone(9, "RTX 4090 phase", "configs/hardware/rtx4090.yaml"),
)


def milestone_status(repo_root: Path) -> list[tuple[Milestone, bool]]:
    return [(m, (repo_root / m.marker_path).exists()) for m in MILESTONES]
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/pytest tests/dashboard/test_milestones.py -v
```

Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/milestones.py tests/dashboard/test_milestones.py
git commit -m "Add milestone status detection for the dashboard"
```

---

### Task 4: Git commit narrative extraction

**Files:**
- Create: `dashboard/narrative.py`
- Test: `tests/dashboard/test_narrative.py`

**Interfaces:**
- Consumes: nothing beyond the `git` CLI.
- Produces: `CommitEntry` (fields: `sha: str`, `subject: str`, `body: str`, `date: str`), `git_log_entries(repo_root: Path, limit: int = 50) -> list[CommitEntry]`. Task 5 imports both.

- [ ] **Step 1: Write the failing test (against a hermetic temp git repo, not this repo)**

`tests/dashboard/test_narrative.py`:

```python
import subprocess
from pathlib import Path

from dashboard.narrative import git_log_entries


def _init_repo_with_commits(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "a.txt").write_text("one")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "First commit\n\nBody line one\nBody line two"],
        cwd=repo, check=True,
    )
    (repo / "b.txt").write_text("two")
    subprocess.run(["git", "add", "b.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "Second commit"], cwd=repo, check=True)


def test_git_log_entries_parses_subject_and_body(tmp_path: Path):
    _init_repo_with_commits(tmp_path)
    entries = git_log_entries(tmp_path)
    assert len(entries) == 2
    # git log defaults to newest first
    assert entries[0].subject == "Second commit"
    assert entries[0].body == ""
    assert entries[1].subject == "First commit"
    assert "Body line one" in entries[1].body
    assert "Body line two" in entries[1].body


def test_git_log_entries_respects_limit(tmp_path: Path):
    _init_repo_with_commits(tmp_path)
    entries = git_log_entries(tmp_path, limit=1)
    assert len(entries) == 1
    assert entries[0].subject == "Second commit"


def test_git_log_entries_sha_is_short(tmp_path: Path):
    _init_repo_with_commits(tmp_path)
    entries = git_log_entries(tmp_path)
    assert all(len(e.sha) == 7 for e in entries)
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/dashboard/test_narrative.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.narrative'`.

- [ ] **Step 3: Write `dashboard/narrative.py`**

```python
from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

_RECORD_SEP = "\x1e"
_FIELD_SEP = "\x1f"


@dataclasses.dataclass(frozen=True)
class CommitEntry:
    sha: str
    subject: str
    body: str
    date: str


def git_log_entries(repo_root: Path, limit: int = 50) -> list[CommitEntry]:
    fmt = f"%H{_FIELD_SEP}%ad{_FIELD_SEP}%s{_FIELD_SEP}%b{_RECORD_SEP}"
    result = subprocess.run(
        ["git", "log", f"-n{limit}", f"--pretty=format:{fmt}", "--date=short"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    entries: list[CommitEntry] = []
    for record in result.stdout.split(_RECORD_SEP):
        record = record.strip("\n")
        if not record:
            continue
        parts = record.split(_FIELD_SEP)
        if len(parts) != 4:
            continue
        sha, date, subject, body = parts
        entries.append(CommitEntry(sha=sha[:7], subject=subject, body=body.strip(), date=date))
    return entries
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/pytest tests/dashboard/test_narrative.py -v
```

Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/narrative.py tests/dashboard/test_narrative.py
git commit -m "Add git commit narrative extraction for the dashboard"
```

---

### Task 5: HTML rendering, CLI entrypoint, and first real generation

**Files:**
- Create: `dashboard/render.py`
- Create: `dashboard/cli.py`
- Modify: `.gitignore` (add `results/dashboard.html`)
- Test: `tests/dashboard/test_render.py`

**Interfaces:**
- Consumes: `dashboard.trials.TrialReport`, `dashboard.trials.discover_trials` (Task 1); `dashboard.svg_chart.latency_distribution_svg` (Task 2); `dashboard.milestones.Milestone`, `dashboard.milestones.milestone_status` (Task 3); `dashboard.narrative.CommitEntry`, `dashboard.narrative.git_log_entries` (Task 4).
- Produces: `render_dashboard(trials: list[TrialReport], commits: list[CommitEntry], milestones: list[tuple[Milestone, bool]]) -> str`. This is the last task in this plan.

- [ ] **Step 1: Write the failing render tests**

`tests/dashboard/test_render.py`:

```python
from dashboard.milestones import Milestone
from dashboard.narrative import CommitEntry
from dashboard.render import render_dashboard
from dashboard.trials import LoadgenSummary, TrialReport


def test_render_dashboard_empty_inputs_does_not_crash():
    html = render_dashboard(trials=[], commits=[], milestones=[])
    assert "<html" in html.lower() or "<!doctype" in html.lower()
    assert "Trials" in html
    assert "Milestones" in html
    assert "Narrative" in html


def test_render_dashboard_includes_loadgen_trial_stats():
    trial = TrialReport(
        filename="baseline.csv",
        is_loadgen_schema=True,
        row_count=31,
        columns=["latency_sec"],
        loadgen_summary=LoadgenSummary(
            row_count=31, success_rate=1.0, p50_latency_sec=5.5,
            p95_latency_sec=11.4, mean_ttft_sec=0.15,
        ),
        preview_rows=[],
    )
    html = render_dashboard(trials=[trial], commits=[], milestones=[])
    assert "baseline.csv" in html
    assert "31" in html


def test_render_dashboard_includes_generic_trial_table():
    trial = TrialReport(
        filename="eval.csv", is_loadgen_schema=False, row_count=1,
        columns=["task", "accuracy"], loadgen_summary=None,
        preview_rows=[{"task": "qa", "accuracy": 0.82}],
    )
    html = render_dashboard(trials=[trial], commits=[], milestones=[])
    assert "eval.csv" in html
    assert "accuracy" in html


def test_render_dashboard_includes_milestone_names_and_status():
    milestones = [(Milestone(1, "Scaffold", "x"), True), (Milestone(2, "Config module", "y"), False)]
    html = render_dashboard(trials=[], commits=[], milestones=milestones)
    assert "Scaffold" in html
    assert "Config module" in html


def test_render_dashboard_includes_commit_subjects():
    commits = [CommitEntry(sha="abc1234", subject="Add thing", body="", date="2026-08-09")]
    html = render_dashboard(trials=[], commits=commits, milestones=[])
    assert "Add thing" in html
    assert "abc1234" in html
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/dashboard/test_render.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.render'`.

- [ ] **Step 3: Write `dashboard/render.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
.venv/bin/pytest tests/dashboard/test_render.py -v
```

Expected: PASS (5 passed).

- [ ] **Step 5: Write the CLI entrypoint**

`dashboard/cli.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from dashboard.milestones import milestone_status
from dashboard.narrative import git_log_entries
from dashboard.render import render_dashboard
from dashboard.trials import discover_trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the local results dashboard")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", default="results/dashboard.html")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    trials = discover_trials(repo_root / "results" / "raw")
    commits = git_log_entries(repo_root)
    milestones = milestone_status(repo_root)

    html = render_dashboard(trials, commits, milestones)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.write_text(html)
    print(f"wrote dashboard to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Add the generated file to `.gitignore`**

Append to `.gitignore`:

```
# generated by dashboard/cli.py — regenerate on demand
/results/dashboard.html
```

- [ ] **Step 7: Run the full test suite**

```bash
.venv/bin/pytest -v
```

Expected: all tests pass, including the 18 from the previous plan plus this plan's ~21 new ones.

- [ ] **Step 8: Generate the real dashboard and sanity-check it**

```bash
.venv/bin/python -m dashboard.cli
```

Expected: prints `wrote dashboard to <repo>/results/dashboard.html`. Open the file (or `grep` for key strings) and confirm: `baseline.csv` appears under Trials with real stats, all milestones 1-7 show as done and 8-9 as pending, and recent commit subjects appear under Narrative.

- [ ] **Step 9: Commit**

```bash
git add dashboard/render.py dashboard/cli.py .gitignore tests/dashboard/test_render.py
git commit -m "Add dashboard HTML rendering, CLI entrypoint, and first real generation"
```

---

## After this plan

`python -m dashboard.cli` regenerates `results/dashboard.html` any time new
trial data lands, without needing to touch the script itself — the generic
fallback table handles CSV schemas it doesn't specifically recognize yet
(like the LongBench eval CSVs milestone 6 will add), and new milestones can
be tracked by adding a `Milestone` entry with an appropriate marker path.
