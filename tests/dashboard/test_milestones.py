from pathlib import Path

from dashboard.milestones import Milestone, milestone_status


def test_milestone_status_marks_existing_marker_done(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "loader.py").touch()
    statuses = milestone_status(tmp_path)
    by_name = {m.name: done for m, done in statuses}
    assert by_name["Config module"] is True


def test_milestone_status_marks_missing_marker_pending(tmp_path: Path):
    statuses = milestone_status(tmp_path)
    by_name = {m.name: done for m, done in statuses}
    assert by_name["Config module"] is False


def test_milestone_status_preserves_milestone_order(tmp_path: Path):
    statuses = milestone_status(tmp_path)
    numbers = [m.number for m, _done in statuses]
    assert numbers == sorted(numbers)


def test_milestone_is_a_plain_dataclass_with_expected_fields():
    m = Milestone(number=1, name="Test", marker_path="foo.py")
    assert (m.number, m.name, m.marker_path) == (1, "Test", "foo.py")
