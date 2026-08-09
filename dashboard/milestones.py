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
