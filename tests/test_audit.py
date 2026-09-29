import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

from prognosebuch.audit import history_violations, manifest_violations, run_audit
from prognosebuch.forecast import RunInfo, run_forecast
from prognosebuch.storage import forecast_paths
from prognosebuch.timeutil import day_bounds_utc
from tests.conftest import FakeClient, synthetic_prices

ISSUE = date(2026, 10, 7)


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def make_repo(root: Path) -> Path:
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "test")
    p = synthetic_prices(date(2026, 5, 1), ISSUE)
    known = p[p.index < day_bounds_utc(ISSUE)[1]]
    run_forecast(
        root, datetime(2026, 10, 7, 7, 5, tzinfo=UTC), FakeClient(known), RunInfo(None, None)
    )
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "forecast")
    return forecast_paths(root, ISSUE, "naive_weekly", "1")[0]


def test_clean_repo_passes(tmp_path: Path) -> None:
    make_repo(tmp_path)
    assert run_audit(tmp_path) == []


def test_modified_forecast_is_detected_in_worktree_and_history(tmp_path: Path) -> None:
    pq_path = make_repo(tmp_path)
    pq_path.write_bytes(pq_path.read_bytes() + b"x")
    assert any("worktree" in v for v in history_violations(tmp_path))
    assert any("SHA-256" in v for v in manifest_violations(tmp_path))
    git(tmp_path, "commit", "-qam", "tamper")
    assert any(v.endswith("naive_weekly.v1.parquet") for v in history_violations(tmp_path))


def test_deleted_forecast_is_detected(tmp_path: Path) -> None:
    pq_path = make_repo(tmp_path)
    git(tmp_path, "rm", "-q", str(pq_path))
    git(tmp_path, "commit", "-qm", "delete")
    assert any(v.split("\t")[0].endswith("D") for v in history_violations(tmp_path))
    assert any("parquet missing" in v for v in manifest_violations(tmp_path))
