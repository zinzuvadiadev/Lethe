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
