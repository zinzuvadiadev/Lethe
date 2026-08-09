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
