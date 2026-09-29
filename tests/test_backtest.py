"""The backtest must reproduce exactly what the live job would have issued."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from prognosebuch.backtest import run_backtest
from prognosebuch.models.base import InfoSet
from prognosebuch.models.lear import LEAR_V1
from prognosebuch.models.naive import NAIVE_SIMILAR_DAY
from tests.conftest import synthetic_prices

LEAR = replace(LEAR_V1, windows=(130,), error_days=6, min_error_days=4)
NAIVE = replace(NAIVE_SIMILAR_DAY, error_days=10, min_error_days=5)
FIRST, LAST = date(2026, 9, 10), date(2026, 9, 13)


@pytest.fixture(scope="module")
def prices() -> pd.Series:
    return synthetic_prices(date(2026, 1, 1), LAST + timedelta(days=2), seed=11)


def test_backtest_matches_live_predict(prices: pd.Series) -> None:
    res = run_backtest([NAIVE, LEAR], prices, FIRST, LAST, datetime(2026, 9, 30, tzinfo=UTC))
    s = res.scores
    assert (s["status"] == "scored").all()
    for model in (NAIVE, LEAR):
        for h in (1, 2):
            t = LAST
            live = model.predict(InfoSet.cut(t - timedelta(days=h), prices), t)
            bt = s[(s["model"] == model.name) & (s["horizon_days"] == h) & (s["target_date"] == t)]
            np.testing.assert_allclose(bt[["q10", "q50", "q90"]].to_numpy(), live.to_numpy())
    assert res.summary["kind"] == "backtest"
    assert set(res.daily["model"]) == {"naive_similar_day", "lear"}
    assert len(res.daily) == 2 * 2 * 4


def test_backtest_matches_live_for_calibrated_gbm() -> None:
    from prognosebuch.models.gbm import GBM_V1
    from tests.conftest import synthetic_weather, weather_driven_prices

    gbm = replace(GBM_V1, train_from=date(2026, 1, 20), max_iter=30, error_days=5, min_error_days=3)
    w = synthetic_weather(date(2026, 1, 1), date(2026, 9, 20), seed=8)
    p = weather_driven_prices(w, seed=9)
    first, last = date(2026, 9, 15), date(2026, 9, 17)
    res = run_backtest([gbm], p, first, last, datetime(2026, 9, 30, tzinfo=UTC), weather=w)
    s = res.scores
    assert (s["status"] == "scored").all()
    for h in (1, 2):
        live = gbm.predict(InfoSet.cut(last - timedelta(days=h), p, w), last)
        bt = s[(s["horizon_days"] == h) & (s["target_date"] == last)]
        np.testing.assert_allclose(bt[["q10", "q50", "q90"]].to_numpy(), live.to_numpy())


def test_conformalize_restores_coverage() -> None:
    from prognosebuch.models.gbm import conformalize
    from prognosebuch.timeutil import day_slots_utc

    rng = np.random.default_rng(0)
    idx = day_slots_utc(date(2026, 9, 1))
    raw = pd.DataFrame({"q10": -1.0, "q50": 0.0, "q90": 1.0}, index=idx)  # far too narrow
    calib = [(raw, pd.Series(rng.normal(0, 5, len(idx)), index=idx)) for _ in range(30)]
    out = conformalize(raw, calib)
    y = rng.normal(0, 5, 20000)
    cover = np.mean((y >= out["q10"].iloc[0]) & (y <= out["q90"].iloc[0]))
    assert 0.77 < cover < 0.83
    assert (out["q50"] == 0).all()
