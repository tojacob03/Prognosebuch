from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from prognosebuch.models.base import InfoSet, Inputs, LeakError, weather_cutoff_utc
from prognosebuch.models.gbm import GBM_V1, sunday_on_or_before
from prognosebuch.timeutil import day_bounds_utc, day_slots_utc
from tests.conftest import synthetic_weather, weather_driven_prices

GBM = replace(GBM_V1, train_from=date(2026, 1, 20), max_iter=40, error_days=6, min_error_days=4)
ISSUE = date(2026, 9, 16)  # a Wednesday


@pytest.fixture(scope="module")
def data() -> tuple[pd.Series, pd.DataFrame]:
    w = synthetic_weather(date(2026, 1, 1), ISSUE + timedelta(days=4), seed=3)
    return weather_driven_prices(w, seed=4), w


def test_sunday_anchor() -> None:
    assert sunday_on_or_before(date(2026, 9, 16)) == date(2026, 9, 13)
    assert sunday_on_or_before(date(2026, 9, 13)) == date(2026, 9, 13)


def test_infoset_rejects_weather_published_after_the_cutoff(
    data: tuple[pd.Series, pd.DataFrame],
) -> None:
    prices, w = data
    p = prices[prices.index < day_bounds_utc(ISSUE)[1]]
    with pytest.raises(LeakError):
        InfoSet(ISSUE, p, w)
    info = InfoSet.cut(ISSUE, prices, w)
    assert info.weather is not None
    assert info.weather["available_at_utc"].max() <= weather_cutoff_utc(ISSUE)
    # the whole D+1 day is covered at lead 2, and the whole D+2 day at lead 3
    for lead, target in ((2, ISSUE + timedelta(days=1)), (3, ISSUE + timedelta(days=2))):
        valid = info.weather.loc[info.weather["lead_days"] == lead, "valid_utc"]
        assert valid.max() >= day_slots_utc(target)[-1].floor("h")


def test_predict_orders_quantiles_and_uses_weather(data: tuple[pd.Series, pd.DataFrame]) -> None:
    prices, w = data
    info = InfoSet.cut(ISSUE, prices, w)
    for h in (1, 2):
        target = ISSUE + timedelta(days=h)
        out = GBM.predict(info, target)
        assert len(out) == 96 and out.notna().all().all()
        assert ((out["q10"] <= out["q50"]) & (out["q50"] <= out["q90"])).all()
    target = ISSUE + timedelta(days=1)
    out = GBM.predict(info, target)
    actual = prices.reindex(out.index)
    naive = prices.reindex(day_slots_utc(ISSUE)).to_numpy()
    assert np.abs(actual - out["q50"]).mean() < np.abs(actual.to_numpy() - naive).mean()


def test_gbm_ignores_everything_after_the_cutoff(data: tuple[pd.Series, pd.DataFrame]) -> None:
    prices, w = data
    tampered_p = prices.copy()
    tampered_p[tampered_p.index >= day_bounds_utc(ISSUE)[1]] = 9999.0
    tampered_w = w.copy()
    late = tampered_w["available_at_utc"] > weather_cutoff_utc(ISSUE)
    tampered_w.loc[late, ["wind_power", "wind_speed", "solar", "temp"]] = -50.0
    target = ISSUE + timedelta(days=2)
    a = GBM.predict(InfoSet.cut(ISSUE, prices, w), target)
    b = GBM.predict(InfoSet.cut(ISSUE, tampered_p, tampered_w), target)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_series_equal(
        GBM.point_for(Inputs(prices, w), ISSUE, target),
        GBM.point_for(Inputs(tampered_p, tampered_w), ISSUE, target),
    )


def test_missing_weather_is_an_error(data: tuple[pd.Series, pd.DataFrame]) -> None:
    prices, _ = data
    from prognosebuch.models.base import InsufficientDataError

    with pytest.raises(InsufficientDataError):
        GBM.predict(InfoSet.cut(ISSUE, prices), ISSUE + timedelta(days=1))
