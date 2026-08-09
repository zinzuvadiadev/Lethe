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
