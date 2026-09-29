"""Workflow files must parse, and every scheduled job must commit via the tested script."""

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


def load(path: Path) -> dict[Any, Any]:
    data = yaml.safe_load(path.read_text())
    assert isinstance(data, dict)
    return data


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_workflow_parses(path: Path) -> None:
    data = load(path)
    assert "jobs" in data
    for job in data["jobs"].values():
        for step in job["steps"]:
            assert isinstance(step.get("run", ""), str)


@pytest.mark.parametrize("name", ["forecast.yml", "evaluate.yml", "probe.yml"])
def test_data_jobs_commit_via_script(name: str) -> None:
    data = load(ROOT / ".github" / "workflows" / name)
    runs = [s.get("run", "") for j in data["jobs"].values() for s in j["steps"]]
    assert any("scripts/commit-data.sh" in r for r in runs)
    assert not any("git push" in r for r in runs)


def test_action_parses() -> None:
    load(ROOT / ".github" / "actions" / "failure-issue" / "action.yml")
