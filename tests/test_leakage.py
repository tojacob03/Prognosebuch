"""The feature/model pipeline must not see data after the information cutoff.

Changing every price after the end of the issue day, including the target days
themselves, must leave the forecast bit-for-bit unchanged.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from prognosebuch.models.base import InfoSet, weather_cutoff_utc
from prognosebuch.timeutil import day_bounds_utc
from tests.conftest import fast_live_models, synthetic_prices, synthetic_weather

ISSUE = date(2026, 10, 7)


@pytest.mark.parametrize("lm", fast_live_models(), ids=lambda lm: lm.key)
def test_forecast_ignores_everything_after_cutoff(lm) -> None:  # type: ignore[no-untyped-def]
    prices = synthetic_prices(date(2026, 1, 1), ISSUE + timedelta(days=5))
    weather = synthetic_weather(date(2026, 1, 1), ISSUE + timedelta(days=5), seed=1)
    tampered = prices.copy()
    tampered[tampered.index >= day_bounds_utc(ISSUE)[1]] = 9999.0
    tampered_w = weather.copy()
    late = tampered_w["available_at_utc"] > weather_cutoff_utc(ISSUE)
    tampered_w.loc[late, ["wind_power", "wind_speed", "solar", "temp"]] = -50.0

    for h in (1, 2):
        target = ISSUE + timedelta(days=h)
        a = lm.model.predict(InfoSet.cut(ISSUE, prices, weather), target)
        b = lm.model.predict(InfoSet.cut(ISSUE, tampered, tampered_w), target)
        pd.testing.assert_frame_equal(a, b)
        assert (a["q90"] < 9999.0).all()


def test_every_live_model_is_covered_by_this_test() -> None:
    from prognosebuch.registry import LIVE_MODELS

    assert [lm.key for lm in fast_live_models()] == [lm.key for lm in LIVE_MODELS]
