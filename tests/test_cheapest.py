from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prognosebuch.cheapest import (
    WINDOW_SLOTS,
    cheapest_window,
    realized,
    rolling_window_means,
    scenario_matrix,
)
from prognosebuch.evaluate import run_evaluate
from prognosebuch.forecast import RunInfo, run_forecast
from prognosebuch.storage import cheapest_score_path, forecast_paths
from prognosebuch.timeutil import day_bounds_utc, day_slots_utc
from tests.conftest import FakeClient, fast_live_models, seeded_client, synthetic_prices

DAY = date(2026, 10, 8)


def curve(values: np.ndarray, d: date = DAY) -> pd.Series:
    return pd.Series(values, index=day_slots_utc(d))


def test_rolling_window_means() -> None:
    x = np.arange(20, dtype=float)
    m = rolling_window_means(x, 4)
    assert len(m) == 17 and m[0] == pytest.approx(1.5) and m[-1] == pytest.approx(17.5)


def test_clear_valley_is_found_with_high_probability() -> None:
    base = np.full(96, 100.0)
    base[48:60] = 20.0  # 12:00-15:00 local is clearly cheapest
    rng = np.random.default_rng(0)
    errors = [curve(rng.normal(0, 5, 96), DAY - timedelta(days=k + 1)) for k in range(40)]
    w = cheapest_window(curve(base), errors)
    assert w["start_local"].startswith("2026-10-08T12:00")
    assert w["end_local"].startswith("2026-10-08T15:00")
    assert w["p_cheapest"] > 0.9 and w["p_near"] > 0.95
    assert w["n_scenarios"] == 40
    assert w["most_likely"][0]["start_local"] == w["start_local"]


def test_flat_day_gives_low_probability() -> None:
    rng = np.random.default_rng(1)
    errors = [curve(rng.normal(0, 30, 96), DAY - timedelta(days=k + 1)) for k in range(60)]
    w = cheapest_window(curve(np.full(96, 80.0)), errors)
    assert w["p_cheapest"] < 0.3


def test_scenarios_align_by_wall_clock_across_dst() -> None:
    # error day with 100 quarter-hours (autumn switch) applied to a normal day
    err_day = date(2026, 10, 25)
    err = pd.Series(1.0, index=day_slots_utc(err_day))
    m = scenario_matrix(curve(np.zeros(96)), [err])
    assert m.shape == (1, 96) and np.allclose(m, 1.0)


def test_realized() -> None:
    actual = np.full(96, 50.0)
    actual[8:20] = 10.0  # 02:00-05:00 cheapest
    a = curve(actual)
    r = realized(a, a.index[8].isoformat())
    assert r["hit"] and r["near"] and r["regret_eur_mwh"] == 0
    r2 = realized(a, a.index[40].isoformat())
    assert not r2["hit"] and not r2["near"] and r2["regret_eur_mwh"] == pytest.approx(40.0)
    assert len(rolling_window_means(actual)) == 96 - WINDOW_SLOTS + 1


def test_pre_registered_and_scored_end_to_end(tmp_path: Path) -> None:
    models = fast_live_models(live_since=date(2026, 10, 7))
    prices = synthetic_prices(date(2026, 1, 1), date(2026, 10, 9), seed=5)
    issue = date(2026, 10, 7)
    known = prices[prices.index < day_bounds_utc(issue)[1]]
    run_forecast(
        tmp_path,
        datetime(2026, 10, 7, 7, 5, tzinfo=UTC),
        seeded_client(tmp_path, known),
        RunInfo(None, None),
        models,
    )
    import json

    m = json.loads(forecast_paths(tmp_path, issue, "lear", "1")[1].read_text())
    cw = m["cheapest_windows"]
    assert cw["window_hours"] == 3 and set(cw["by_target"]) == {"2026-10-08", "2026-10-09"}
    assert 0 <= cw["by_target"]["2026-10-08"]["p_cheapest"] <= 1

    visible = prices[prices.index < day_bounds_utc(date(2026, 10, 8))[1]]
    run_evaluate(tmp_path, datetime(2026, 10, 7, 14, 0, tzinfo=UTC), FakeClient(visible), models)
    c = pd.read_parquet(cheapest_score_path(tmp_path, date(2026, 10, 8)))
    assert set(c["status"]) == {"scored"} and len(c) == len(models)
    assert (c["regret_eur_mwh"] >= 0).all()
    summary = json.loads((tmp_path / "scores" / "summary.json").read_text())
    rows = summary["cheapest_windows"]["windows"]["all"]
    assert {r["model"] for r in rows} == {lm.key for lm in models}
