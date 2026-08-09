from pathlib import Path

from dashboard.milestones import Milestone, milestone_status


def test_milestone_status_marks_existing_marker_done(tmp_path: Path):
    (tmp_path / "serving").mkdir()
    (tmp_path / "serving" / "server.py").touch()
    statuses = milestone_status(tmp_path)
    by_name = {m.name: done for m, done in statuses}
    assert by_name["Baseline vLLM serving"] is True


def test_milestone_status_marks_missing_marker_pending(tmp_path: Path):
    statuses = milestone_status(tmp_path)
    by_name = {m.name: done for m, done in statuses}
    assert by_name["Baseline vLLM serving"] is False


def test_milestone_status_preserves_milestone_order(tmp_path: Path):
    statuses = milestone_status(tmp_path)
    numbers = [m.number for m, _done in statuses]
    assert numbers == sorted(numbers)


def test_milestone_is_a_plain_dataclass_with_expected_fields():
    m = Milestone(number=1, name="Test", marker_path="foo.py")
    assert (m.number, m.name, m.marker_path) == (1, "Test", "foo.py")


def test_milestone_with_placeholder_text_pending_while_placeholder_present(tmp_path: Path):
    (tmp_path / "README.md").write_text("# Title\n\n_Filled in at milestone 8._\n")
    statuses = milestone_status(tmp_path)
    by_name = {m.name: done for m, done in statuses}
    assert by_name["README"] is False


def test_milestone_with_placeholder_text_done_once_placeholder_removed(tmp_path: Path):
    (tmp_path / "README.md").write_text("# Title\n\nReal content, no placeholders here.\n")
    statuses = milestone_status(tmp_path)
    by_name = {m.name: done for m, done in statuses}
    assert by_name["README"] is True
