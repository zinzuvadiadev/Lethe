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
