from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from prognosebuch.models.base import InfoSet, InsufficientDataError, LeakError
from prognosebuch.models.naive import (
    NAIVE_LAST_DAY,
    NAIVE_SIMILAR_DAY,
    NAIVE_WEEKLY,
    NaiveModel,
    profile_forecast,
)
from prognosebuch.timeutil import day_slots_utc, wall_clock
from tests.conftest import synthetic_prices


def test_profile_normal_to_autumn_switch_repeats_hour(real_prices: pd.Series) -> None:
    fc = profile_forecast(real_prices, date(2025, 10, 19), date(2025, 10, 26))
    assert len(fc) == 100
    labels = wall_clock(pd.DatetimeIndex(fc.index))
    twice = fc[np.asarray(labels == "02:30")]
    assert len(twice) == 2 and twice.iloc[0] == twice.iloc[1]


def test_profile_autumn_switch_source_averages_doubled_hour(real_prices: pd.Series) -> None:
    fc = profile_forecast(real_prices, date(2025, 10, 26), date(2025, 10, 27))
    src = real_prices.reindex(day_slots_utc(date(2025, 10, 26)))
    src_0230 = src[np.asarray(wall_clock(pd.DatetimeIndex(src.index)) == "02:30")]
    got = fc[np.asarray(wall_clock(pd.DatetimeIndex(fc.index)) == "02:30")]
    assert len(fc) == 96
    assert got.iloc[0] == pytest.approx(src_0230.mean())


def test_profile_spring_switch_both_directions(real_prices: pd.Series) -> None:
    to_spring = profile_forecast(real_prices, date(2026, 3, 22), date(2026, 3, 29))
    assert len(to_spring) == 92 and to_spring.notna().all()
    from_spring = profile_forecast(real_prices, date(2026, 3, 29), date(2026, 3, 30))
    assert len(from_spring) == 96 and from_spring.notna().all()
    # the missing 02:xx hour is filled with the previous quarter-hour (01:45)
    labels = wall_clock(pd.DatetimeIndex(from_spring.index))
    assert from_spring[np.asarray(labels == "02:00")].iloc[0] == pytest.approx(
        from_spring[np.asarray(labels == "01:45")].iloc[0]
    )


def test_negative_prices_pass_through(real_prices: pd.Series) -> None:
    neg_day = (
        pd.DatetimeIndex(real_prices[real_prices < 0].index).tz_convert("Europe/Berlin")[0].date()
    )
    fc = profile_forecast(real_prices, neg_day, neg_day + timedelta(days=1))
    assert (fc < 0).any()


def test_incomplete_source_day_raises() -> None:
    p = synthetic_prices(date(2026, 1, 1), date(2026, 1, 10))
    p = p.drop(
        p.index[pd.DatetimeIndex(p.index).tz_convert("Europe/Berlin").date == date(2026, 1, 5)][:20]
    )
    with pytest.raises(InsufficientDataError):
        profile_forecast(p, date(2026, 1, 5), date(2026, 1, 6))


def test_small_gaps_are_filled() -> None:
    p = synthetic_prices(date(2026, 1, 1), date(2026, 1, 10))
    day = p.index[pd.DatetimeIndex(p.index).tz_convert("Europe/Berlin").date == date(2026, 1, 5)]
    p.loc[day[10:13]] = np.nan
    fc = profile_forecast(p.dropna(), date(2026, 1, 5), date(2026, 1, 6))
    assert fc.notna().all()


@pytest.mark.parametrize("model", [NAIVE_LAST_DAY, NAIVE_WEEKLY, NAIVE_SIMILAR_DAY])
@pytest.mark.parametrize("horizon", [1, 2])
def test_source_day_is_never_after_issue_day(model: NaiveModel, horizon: int) -> None:
    for k in range(14):  # every weekday
        issue = date(2026, 10, 1) + timedelta(days=k)
        assert model.source_day(issue, issue + timedelta(days=horizon)) <= issue


def test_similar_day_rule() -> None:
    # Thursday issue, Friday target (D+1): use the day before = Thursday.
    assert NAIVE_SIMILAR_DAY.source_day(date(2026, 10, 1), date(2026, 10, 2)) == date(2026, 10, 1)
    # Thursday issue, Saturday target: one week before.
    assert NAIVE_SIMILAR_DAY.source_day(date(2026, 10, 1), date(2026, 10, 3)) == date(2026, 9, 26)
    # Tuesday issue, Thursday target (D+2): the day before is unknown -> last known day.
    assert NAIVE_SIMILAR_DAY.source_day(date(2026, 9, 29), date(2026, 10, 1)) == date(2026, 9, 29)


@pytest.mark.parametrize("model", [NAIVE_LAST_DAY, NAIVE_WEEKLY, NAIVE_SIMILAR_DAY])
def test_predict_quantiles_ordered_on_real_data(model: NaiveModel, real_prices: pd.Series) -> None:
    m = replace(model, window_days=14, min_error_days=7)
    for issue in (date(2025, 10, 25), date(2025, 10, 26), date(2026, 4, 3), date(2026, 3, 28)):
        info = InfoSet.cut(issue, real_prices)
        for h in (1, 2):
            out = m.predict(info, issue + timedelta(days=h))
            assert out.notna().all().all()
            assert (out["q10"] <= out["q50"]).all() and (out["q50"] <= out["q90"]).all()
            assert len(out) == len(day_slots_utc(issue + timedelta(days=h)))


def test_too_little_history_raises() -> None:
    p = synthetic_prices(date(2026, 9, 1), date(2026, 9, 10))
    with pytest.raises(InsufficientDataError):
        NAIVE_WEEKLY.predict(InfoSet.cut(date(2026, 9, 10), p), date(2026, 9, 11))


def test_infoset_rejects_future_prices() -> None:
    p = synthetic_prices(date(2026, 9, 1), date(2026, 9, 12))
    with pytest.raises(LeakError):
        InfoSet(issue_date=date(2026, 9, 10), prices=p)
    assert (
        InfoSet.cut(date(2026, 9, 10), p).prices.index.max() < day_slots_utc(date(2026, 9, 11))[0]
    )
