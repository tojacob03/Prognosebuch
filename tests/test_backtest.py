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
