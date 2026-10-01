"""The release workflow must not run its publishing steps twice for the same day."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_release_steps_are_guarded_and_use_one_tag_variable() -> None:
    data = yaml.safe_load((ROOT / ".github" / "workflows" / "release.yml").read_text())
    steps = data["jobs"]["release"]["steps"]
    guarded = [s for s in steps if "if" in s and "exists.outputs.skip" in s["if"]]
    names = {s["name"] for s in guarded}
    assert {"Audit before publishing", "Build archive and notes", "Publish release"} <= names
    text = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "$tag" not in text.split("Build archive", 1)[1]  # only $TAG after the first step
    assert data["concurrency"]["group"] == "release"


def test_zenodo_metadata_is_complete() -> None:
    import json

    z = json.loads((ROOT / ".zenodo.json").read_text())
    assert z["upload_type"] == "dataset" and z["license"] == "cc-by-4.0"
    assert z["creators"][0]["name"] and z["title"] and z["description"]
