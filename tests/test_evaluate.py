import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from prognosebuch.evaluate import load_actuals, run_evaluate
from prognosebuch.forecast import RunInfo, run_forecast
from prognosebuch.registry import LIVE_MODELS, LiveModel
from prognosebuch.storage import score_path
from prognosebuch.timeutil import day_bounds_utc
from tests.conftest import FakeClient, synthetic_prices

LIVE = date(2026, 10, 7)
MODELS = tuple(LiveModel(lm.model, live_since=LIVE) for lm in LIVE_MODELS)
ALL = synthetic_prices(date(2026, 5, 1), date(2026, 10, 12), seed=3)


def issue(root: Path, d: date) -> None:
    known = ALL[ALL.index < day_bounds_utc(d)[1]]
    now = datetime(d.year, d.month, d.day, 7, 5, tzinfo=UTC)
    run_forecast(root, now, FakeClient(known), RunInfo(None, None), MODELS)


def evaluate(root: Path, until: date, now: datetime) -> None:
    visible = ALL[ALL.index < day_bounds_utc(until)[1]]
    run_evaluate(root, now, FakeClient(visible), MODELS)


def test_scores_and_missed_days(tmp_path: Path) -> None:
    issue(tmp_path, date(2026, 10, 7))
    # 2026-10-08: job failed, no forecast -> must show up as missed
    evaluate(tmp_path, date(2026, 10, 10), datetime(2026, 10, 9, 12, 0, tzinfo=UTC))

    s8 = pd.read_parquet(score_path(tmp_path, date(2026, 10, 8)))
    assert set(s8["horizon_days"]) == {1}  # the D+2 forecast was due only from 2026-10-07
    assert (s8["status"] == "scored").all() and len(s8) == 3 * 96

    s9 = pd.read_parquet(score_path(tmp_path, date(2026, 10, 9)))
    st = s9.groupby("horizon_days")["status"].unique().to_dict()
    assert list(st[1]) == ["missed"] and list(st[2]) == ["scored"]
    assert s9.loc[s9["status"] == "missed", "q50"].isna().all()
    assert s9["actual"].notna().all()

    summary = json.loads((tmp_path / "scores" / "summary.json").read_text())
    missed = {(m["model"], m["horizon_days"], m["target_date"]) for m in summary["missed"]}
    assert ("naive_weekly.v1", 1, "2026-10-09") in missed
    rows = {(r["model"], r["horizon_days"]): r for r in summary["windows"]["all"]["models"]}
    ref = rows[("naive_similar_day.v1", 1)]
    assert ref["all"]["skill_mae"] == 0.0
    assert ref["days_due"] == 3 and ref["days_forecast"] == 1
    assert 0 <= ref["all"]["coverage_80"] <= 1


def test_evaluate_is_idempotent_and_waits_for_complete_prices(tmp_path: Path) -> None:
    issue(tmp_path, date(2026, 10, 7))
    evaluate(tmp_path, date(2026, 10, 7), datetime(2026, 10, 7, 12, 0, tzinfo=UTC))
    assert not score_path(tmp_path, date(2026, 10, 8)).exists()  # prices not yet published

    evaluate(tmp_path, date(2026, 10, 8), datetime(2026, 10, 7, 12, 0, tzinfo=UTC))
    p = score_path(tmp_path, date(2026, 10, 8))
    first = p.read_bytes()
    evaluate(tmp_path, date(2026, 10, 8), datetime(2026, 10, 7, 16, 0, tzinfo=UTC))
    assert p.read_bytes() == first


def test_actuals_archive_roundtrip(tmp_path: Path) -> None:
    evaluate(tmp_path, date(2026, 10, 8), datetime(2026, 10, 7, 12, 0, tzinfo=UTC))
    a = load_actuals(tmp_path)
    assert a.index.max() == day_bounds_utc(date(2026, 10, 8))[1] - pd.Timedelta(minutes=15)
    expected = ALL[(ALL.index >= a.index.min()) & (ALL.index <= a.index.max())]
    assert (a - expected).abs().max() < 1e-9
    assert a.index.min() >= day_bounds_utc(date(2026, 10, 7) - timedelta(days=21))[0]
