"""Checks that make the book verifiable.

1. Git history: no file under ``forecasts/`` was ever modified, deleted or renamed
   (including uncommitted changes in the working tree and index).
2. Every forecast parquet has a manifest, and the manifest's SHA-256 matches the file.
3. Every forecast was issued inside the allowed window of its issue date.
"""

from __future__ import annotations

import json
import subprocess
from datetime import date, datetime
from pathlib import Path

from prognosebuch.forecast import DEADLINE, EARLIEST_ISSUE
from prognosebuch.storage import sha256_hex
from prognosebuch.timeutil import TZ

FORECASTS = "forecasts"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def history_violations(root: Path) -> list[str]:
    """Modified/deleted/type-changed forecast files in history, index or working tree."""
    out: list[str] = []
    has_head = (
        subprocess.run(
            ["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=root, capture_output=True
        ).returncode
        == 0
    )
    if has_head:
        log = _git(
            root,
            "log",
            "-m",
            "--no-renames",
            "--diff-filter=MDT",
            "--name-status",
            "--format=commit %H",
            "--",
            FORECASTS,
        )
        commit = ""
        for line in log.splitlines():
            if line.startswith("commit "):
                commit = line.split()[1][:12]
            elif line.strip():
                out.append(f"{commit}: {line.strip()}")
        for label, args in (
            ("index", ("diff", "--cached", "--no-renames", "--name-status", "HEAD")),
            ("worktree", ("diff", "--no-renames", "--name-status")),
        ):
            for line in _git(root, *args, "--diff-filter=MDT", "--", FORECASTS).splitlines():
                if line.strip():
                    out.append(f"{label}: {line.strip()}")
    return out


def manifest_violations(root: Path) -> list[str]:
    out: list[str] = []
    base = root / FORECASTS
    if not base.exists():
        return out
    for pq_path in sorted(base.rglob("*.parquet")):
        js_path = pq_path.with_suffix(".json")
        rel = pq_path.relative_to(root)
        if not js_path.exists():
            out.append(f"{rel}: manifest missing")
            continue
        manifest = json.loads(js_path.read_text())
        if manifest.get("parquet_sha256") != sha256_hex(pq_path.read_bytes()):
            out.append(f"{rel}: SHA-256 does not match manifest")
        issue_date = date.fromisoformat(manifest["issue_date"])
        if pq_path.parent.name != issue_date.isoformat():
            out.append(f"{rel}: directory does not match issue_date {issue_date}")
        issued = datetime.fromisoformat(manifest["issued_at_utc"]).astimezone(TZ)
        if issued.date() != issue_date or not (EARLIEST_ISSUE <= issued.time() < DEADLINE):
            out.append(f"{rel}: issued at {issued.isoformat()}, outside the allowed window")
    for js_path in sorted(base.rglob("*.json")):
        if not js_path.with_suffix(".parquet").exists():
            out.append(f"{js_path.relative_to(root)}: parquet missing")
    return out


def run_audit(root: Path) -> list[str]:
    return history_violations(root) + manifest_violations(root)
