from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Milestone:
    number: int
    name: str
    marker_path: str
    # If set, marker_path must exist AND NOT contain this text to count as
    # done. Used for milestones (like README) whose marker file exists from
    # the initial scaffold as a placeholder and only becomes "done" once
    # real content replaces it.
    placeholder_text: str | None = None


# Mirrors design doc §11 (docs/superpowers/specs/2026-08-09-kv-cache-eviction-
# benchmark-design.md) exactly — the 9 real project milestones, not the
# finer-grained per-plan task breakdown.
MILESTONES: tuple[Milestone, ...] = (
    Milestone(1, "Scaffold", "pytest.ini"),
    Milestone(2, "Baseline vLLM serving", "serving/server.py"),
    Milestone(3, "Load generator vs. baseline", "results/raw/baseline.csv"),
    Milestone(4, "Eviction policy implemented", "serving/eviction.py"),
    Milestone(5, "Sweep across aggressiveness", "results/raw/aggressiveness_sweep.csv"),
    Milestone(6, "LongBench subset eval harness", "eval/harness.py"),
    Milestone(7, "Combine into tradeoff plot", "results/tradeoff_plot.png"),
    Milestone(8, "README", "README.md", placeholder_text="_Filled in at milestone"),
    Milestone(9, "Phase 2 (later): RTX 4090", "configs/hardware/rtx4090.yaml"),
)


def milestone_status(repo_root: Path) -> list[tuple[Milestone, bool]]:
    results: list[tuple[Milestone, bool]] = []
    for m in MILESTONES:
        path = repo_root / m.marker_path
        if not path.exists():
            done = False
        elif m.placeholder_text is not None:
            done = m.placeholder_text not in path.read_text()
        else:
            done = True
        results.append((m, done))
    return results
