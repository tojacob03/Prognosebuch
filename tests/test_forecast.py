import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from prognosebuch.forecast import (
    DeadlinePassedError,
    PricesAlreadyPublishedError,
    RunInfo,
    TooEarlyError,
    run_forecast,
)
from prognosebuch.storage import forecast_paths, sha256_hex
from prognosebuch.timeutil import day_bounds_utc
from tests.conftest import FakeClient, fast_live_models, seeded_client, synthetic_prices

ISSUE = date(2026, 10, 7)
RUN = RunInfo(code_commit="abc123", workflow_run_url="https://example.invalid/run/1")
MODELS = fast_live_models()
N = len(MODELS)


def known_until_issue_day() -> pd.Series:
    p = synthetic_prices(date(2026, 1, 1), ISSUE)
    return p[p.index < day_bounds_utc(ISSUE)[1]]


def at(hh: int, mm: int, d: date = ISSUE) -> datetime:
    # October 2026 before the switch: CEST = UTC+2
    return datetime(d.year, d.month, d.day, hh - 2, mm, tzinfo=UTC)


def test_writes_immutable_files_and_manifest(tmp_path: Path) -> None:
    res = run_forecast(
        tmp_path, at(9, 5), seeded_client(tmp_path, known_until_issue_day()), RUN, MODELS
    )
    assert not res.failed
    assert len(res.written) == 2 * N
    pq_path, js_path = forecast_paths(tmp_path, ISSUE, "naive_weekly", "1")
    df = pd.read_parquet(pq_path)
    assert set(df["horizon_days"]) == {1, 2}
    assert len(df) == 96 * 2
    m = json.loads(js_path.read_text())
    assert m["parquet_sha256"] == sha256_hex(pq_path.read_bytes())
    assert m["issued_at_local"].startswith("2026-10-07T09:05")
    assert m["code_commit"] == "abc123"
    cut = m["data_cutoffs"]["smard:4169:DE-LU:quarterhour"]
    assert cut["last_value_used_utc"] == "2026-10-07T21:45:00+00:00"


def test_second_run_is_a_no_op(tmp_path: Path) -> None:
    client = seeded_client(tmp_path, known_until_issue_day())
    run_forecast(tmp_path, at(9, 5), client, RUN, MODELS)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    res = run_forecast(tmp_path, at(10, 5), client, RUN, MODELS)
    assert res.written == [] and len(res.skipped_existing) == N
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_missing_model_is_added_later_without_touching_others(tmp_path: Path) -> None:
    client = seeded_client(tmp_path, known_until_issue_day())
    run_forecast(tmp_path, at(9, 5), client, RUN, MODELS)
    pq_path, js_path = forecast_paths(tmp_path, ISSUE, "naive_last_day", "1")
    pq_path.unlink()
    js_path.unlink()
    res = run_forecast(tmp_path, at(9, 40), client, RUN, MODELS)
    assert res.written == [pq_path, js_path]


def test_too_early_and_deadline(tmp_path: Path) -> None:
    client = seeded_client(tmp_path, known_until_issue_day())
    with pytest.raises(TooEarlyError):
        run_forecast(tmp_path, at(8, 30), client, RUN, MODELS)
    with pytest.raises(DeadlinePassedError):
        run_forecast(tmp_path, at(12, 0), client, RUN, MODELS)
    assert not any((tmp_path / "forecasts").rglob("*.parquet"))


def test_winter_time_window(tmp_path: Path) -> None:
    issue = date(2026, 11, 10)  # CET = UTC+1
    p = synthetic_prices(date(2026, 1, 1), issue)
    client = seeded_client(tmp_path, p[p.index < day_bounds_utc(issue)[1]])
    with pytest.raises(TooEarlyError):  # 07:50 UTC = 08:50 CET is fine, 07:40 UTC is not
        run_forecast(tmp_path, datetime(2026, 11, 10, 7, 40, tzinfo=UTC), client, RUN, MODELS)
    res = run_forecast(tmp_path, datetime(2026, 11, 10, 7, 50, tzinfo=UTC), client, RUN, MODELS)
    assert len(res.written) == 2 * N


def test_refuses_when_next_day_prices_are_already_public(tmp_path: Path) -> None:
    p = synthetic_prices(date(2026, 5, 1), ISSUE + timedelta(days=1))
    with pytest.raises(PricesAlreadyPublishedError):
        run_forecast(tmp_path, at(9, 5), FakeClient(p), RUN, MODELS)
    assert not any((tmp_path / "forecasts").rglob("*.parquet"))


def test_autumn_dst_target_has_100_rows(tmp_path: Path) -> None:
    issue = date(2026, 10, 24)
    p = synthetic_prices(date(2026, 1, 1), issue)
    client = seeded_client(tmp_path, p[p.index < day_bounds_utc(issue)[1]])
    run_forecast(tmp_path, datetime(2026, 10, 24, 7, 5, tzinfo=UTC), client, RUN, MODELS)
    df = pd.read_parquet(forecast_paths(tmp_path, issue, "naive_last_day", "1")[0])
    sizes = df.groupby("target_date").size()
    assert sizes[date(2026, 10, 25)] == 100 and sizes[date(2026, 10, 26)] == 96


def test_partial_failure_keeps_other_models(tmp_path: Path) -> None:
    # Only 20 days of history: the weekly models cannot estimate their error bands.
    p = synthetic_prices(ISSUE - timedelta(days=20), ISSUE)
    res = run_forecast(
        tmp_path, at(9, 5), FakeClient(p[p.index < day_bounds_utc(ISSUE)[1]]), RUN, MODELS
    )
    assert set(res.failed) == {lm.key for lm in MODELS}
    assert res.written == []
