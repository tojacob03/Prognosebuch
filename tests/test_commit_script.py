"""The shell script every scheduled job uses to commit its data."""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "commit-data.sh"


def run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PUSH": "0"}
    return subprocess.run(
        ["bash", str(SCRIPT), *args], cwd=root, env=env, capture_output=True, text=True
    )


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "README.md").write_text("x\n")
    git(tmp_path, "add", "README.md")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@e.invalid", "commit", "-qm", "init")
    return tmp_path


def test_missing_paths_are_skipped_and_nothing_to_commit(tmp_path: Path) -> None:
    root = repo(tmp_path)
    r = run(root, "evaluate: test", "actuals", "scores")
    assert r.returncode == 0, r.stderr
    assert "nothing to commit" in r.stdout


def test_new_files_are_committed_even_if_other_paths_are_missing(tmp_path: Path) -> None:
    root = repo(tmp_path)
    (root / "actuals").mkdir()
    (root / "actuals" / "a.parquet").write_bytes(b"1")
    r = run(root, "evaluate: test", "actuals", "scores")
    assert r.returncode == 0, r.stderr
    assert git(root, "log", "-1", "--format=%s").strip() == "evaluate: test"
    assert "actuals/a.parquet" in git(root, "show", "--name-only", "--format=")


def test_changing_an_existing_forecast_is_refused(tmp_path: Path) -> None:
    root = repo(tmp_path)
    f = root / "forecasts" / "2026" / "10" / "2026-10-01" / "m.v1.parquet"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"original")
    assert run(root, "forecast: day 1", "forecasts").returncode == 0
    f.write_bytes(b"tampered")
    r = run(root, "forecast: tamper", "forecasts")
    assert r.returncode == 1
    assert "refusing" in r.stdout
    assert git(root, "log", "-1", "--format=%s").strip() == "forecast: day 1"
