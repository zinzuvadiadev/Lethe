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
