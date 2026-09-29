from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from prognosebuch.models.base import InfoSet, InsufficientDataError
from prognosebuch.models.lear import LEAR_V1, design, hourly_matrix, is_holiday
from prognosebuch.timeutil import day_bounds_utc, day_slots_utc
from tests.conftest import synthetic_prices

SMALL = replace(LEAR_V1, windows=(130,), error_days=8, min_error_days=5)
ISSUE = date(2026, 10, 20)


@pytest.fixture(scope="module")
def prices() -> pd.Series:
    return synthetic_prices(date(2026, 1, 1), ISSUE + timedelta(days=3), seed=7)


def test_hourly_matrix_handles_dst_days(real_prices: pd.Series) -> None:
    m = hourly_matrix(real_prices)
    assert [int(c) for c in m.columns] == list(range(24))
    for d in (date(2025, 10, 26), date(2026, 3, 29)):
        assert m.loc[d].notna().all()
    # autumn: the doubled 02:00 hour is averaged
    day = real_prices.reindex(day_slots_utc(date(2025, 10, 26)))
    two = day[np.asarray(pd.DatetimeIndex(day.index).tz_convert("Europe/Berlin").hour == 2)]
    assert m.loc[date(2025, 10, 26), 2] == pytest.approx(two.mean())


def test_design_uses_only_lagged_days(prices: pd.Series) -> None:
    daily = hourly_matrix(prices)
    X, _ = design(daily, 2, until=ISSUE + timedelta(days=2))
    t = ISSUE + timedelta(days=2)
    assert X.loc[t, "lag2_h05"] == pytest.approx(daily.loc[ISSUE, 5])
    assert X.loc[t, "lag7_h23"] == pytest.approx(daily.loc[t - timedelta(days=7), 23])
    assert not any(c.startswith(("lag0", "lag1_")) for c in X.columns)


def test_holidays() -> None:
    assert is_holiday(date(2026, 10, 3)) and is_holiday(date(2026, 12, 25))
    assert not is_holiday(date(2026, 10, 5))


def test_predict_shapes_and_order(prices: pd.Series) -> None:
    info = InfoSet.cut(ISSUE, prices)
    for h in (1, 2):
        out = SMALL.predict(info, ISSUE + timedelta(days=h))
        assert len(out) == len(day_slots_utc(ISSUE + timedelta(days=h)))
        assert out.notna().all().all()
        assert ((out["q10"] <= out["q50"]) & (out["q50"] <= out["q90"])).all()


def test_learns_the_daily_shape(prices: pd.Series) -> None:
    info = InfoSet.cut(ISSUE, prices)
    target = ISSUE + timedelta(days=1)
    out = SMALL.predict(info, target)
    actual = prices.reindex(out.index)
    naive = prices.reindex(day_slots_utc(ISSUE)).to_numpy()
    assert np.abs(actual - out["q50"]).mean() < np.abs(actual.to_numpy() - naive).mean()


def test_lear_ignores_data_after_cutoff(prices: pd.Series) -> None:
    tampered = prices.copy()
    tampered[tampered.index >= day_bounds_utc(ISSUE)[1]] = -5000.0
    target = ISSUE + timedelta(days=2)
    a = SMALL.predict(InfoSet.cut(ISSUE, prices), target)
    b = SMALL.predict(InfoSet.cut(ISSUE, tampered), target)
    pd.testing.assert_frame_equal(a, b)
    # point_for must cut by itself as well, even when handed the full series
    pd.testing.assert_series_equal(
        SMALL.point_for(prices, ISSUE, target), SMALL.point_for(tampered, ISSUE, target)
    )


def test_not_enough_history() -> None:
    p = synthetic_prices(date(2026, 9, 1), ISSUE)
    with pytest.raises(InsufficientDataError):
        SMALL.predict(InfoSet.cut(ISSUE, p), ISSUE + timedelta(days=1))
