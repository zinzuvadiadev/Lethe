import csv
from pathlib import Path

from loadgen.client import RequestResult
from loadgen.runner import CSV_FIELDS, SweepConfig, result_to_row, write_csv

CFG = SweepConfig(
    base_url="http://localhost:8000",
    served_model_name="qwen3-4b-instruct-2507",
    rate_per_sec=2.0,
    duration_sec=10.0,
    eviction_aggressiveness="baseline",
    slo_latency_sec=2.0,
)


def test_result_to_row_marks_slo_met_when_within_latency():
    result = RequestResult(100, 50, 50, 1.5, 0.2, True, None)
    row = result_to_row(result, CFG)
    assert row["met_slo"] is True
    assert row["success"] is True


def test_result_to_row_marks_slo_missed_when_over_latency():
    result = RequestResult(100, 50, 50, 3.0, 0.2, True, None)
    row = result_to_row(result, CFG)
    assert row["met_slo"] is False


def test_result_to_row_failed_request_never_meets_slo():
    result = RequestResult(100, 50, 0, 0.1, None, False, "boom")
    row = result_to_row(result, CFG)
    assert row["met_slo"] is False


def test_write_csv_creates_file_with_header_and_rows(tmp_path: Path):
    rows = [result_to_row(RequestResult(100, 50, 50, 1.0, 0.1, True, None), CFG)]
    out = tmp_path / "nested" / "out.csv"
    write_csv(rows, out)
    with out.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_FIELDS
        read_rows = list(reader)
    assert len(read_rows) == 1
    assert read_rows[0]["eviction_aggressiveness"] == "baseline"
