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


def test_render_dashboard_shows_milestone_progress_summary():
    milestones = [(Milestone(1, "Scaffold", "x"), True), (Milestone(2, "Config module", "y"), False)]
    html = render_dashboard(trials=[], commits=[], milestones=milestones)
    assert "1 of 2 milestones complete" in html


def test_render_dashboard_pill_reflects_high_success_rate():
    trial = TrialReport(
        filename="good.csv", is_loadgen_schema=True, row_count=10, columns=[],
        loadgen_summary=LoadgenSummary(
            row_count=10, success_rate=1.0, p50_latency_sec=1.0,
            p95_latency_sec=2.0, mean_ttft_sec=0.1,
        ),
        preview_rows=[],
    )
    html = render_dashboard(trials=[trial], commits=[], milestones=[])
    assert "pill-success" in html


def test_render_dashboard_pill_reflects_low_success_rate():
    trial = TrialReport(
        filename="bad.csv", is_loadgen_schema=True, row_count=10, columns=[],
        loadgen_summary=LoadgenSummary(
            row_count=10, success_rate=0.5, p50_latency_sec=1.0,
            p95_latency_sec=2.0, mean_ttft_sec=0.1,
        ),
        preview_rows=[],
    )
    html = render_dashboard(trials=[trial], commits=[], milestones=[])
    assert "pill-danger" in html


def test_render_dashboard_includes_generated_at_when_provided():
    html = render_dashboard(trials=[], commits=[], milestones=[], generated_at="2026-08-10 12:00 UTC")
    assert "2026-08-10 12:00 UTC" in html


def test_render_dashboard_omits_generated_at_when_not_provided():
    html = render_dashboard(trials=[], commits=[], milestones=[])
    assert '<p class="generated-at">' not in html
