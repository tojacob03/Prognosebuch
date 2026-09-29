"""Naive benchmark models ("rules of thumb").

Each rule copies the price profile of one fully known past day onto the target day,
matched by local wall-clock time. The median (q50) is the pure rule. The 80 % band
(q10, q90) adds empirical error quantiles of the same rule over the previous
``window_days`` days, per local hour and horizon, so that pinball loss and band
coverage are comparable with the real models.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from prognosebuch.models.bands import predict_with_bands
from prognosebuch.models.base import InfoSet, InsufficientDataError, LeakError
from prognosebuch.timeutil import day_slots_utc, wall_clock

MIN_SOURCE_COVERAGE = 0.9


def profile_forecast(prices: pd.Series, source: date, target: date) -> pd.Series:
    """Map the price profile of ``source`` onto the quarter-hours of ``target``.

    Matching is by local wall clock, so DST days work: a repeated hour (autumn) takes
    the value of the single hour on a normal day, a doubled source hour is averaged, and
    a missing source hour (spring) is filled with the previous quarter-hour.
    """
    src_slots = day_slots_utc(source)
    src = prices.reindex(src_slots)
    if src.notna().mean() < MIN_SOURCE_COVERAGE:
        raise InsufficientDataError(f"prices for source day {source} are incomplete")
    by_clock = src.groupby(wall_clock(src_slots)).mean()
    tgt_slots = day_slots_utc(target)
    vals = by_clock.reindex(wall_clock(tgt_slots)).to_numpy()
    return pd.Series(vals, index=tgt_slots, name="q50").ffill().bfill()


@dataclass(frozen=True)
class NaiveModel:
    name: str
    version: str
    description: str
    rule: str  # "last_day" | "weekly" | "similar_day"
    error_days: int = 90
    min_error_days: int = 30

    def source_day(self, issue_date: date, target_date: date) -> date:
        if self.rule == "last_day":
            src = issue_date
        elif self.rule == "weekly":
            src = target_date - timedelta(days=7)
        elif self.rule == "similar_day":
            # Lago et al. (2021): Mon/Sat/Sun from one week before, Tue-Fri from the day
            # before. For D+2 the day before is not yet known, so the last known day is used.
            if target_date.weekday() in (0, 5, 6):
                src = target_date - timedelta(days=7)
            else:
                src = min(target_date - timedelta(days=1), issue_date)
        else:
            raise ValueError(f"unknown rule {self.rule}")
        if src > issue_date:
            raise LeakError(f"{self.name}: source day {src} is after issue day {issue_date}")
        return src

    def point_for(self, prices: pd.Series, issue: date, target: date) -> pd.Series:
        return profile_forecast(prices, self.source_day(issue, target), target)

    def predict(self, info: InfoSet, target_date: date) -> pd.DataFrame:
        return predict_with_bands(self, info, target_date)


NAIVE_LAST_DAY = NaiveModel(
    name="naive_last_day",
    version="1",
    description="Same quarter-hour on the last known day (the issue day).",
    rule="last_day",
)
NAIVE_WEEKLY = NaiveModel(
    name="naive_weekly",
    version="1",
    description="Same quarter-hour one week before the target day (seasonal naive).",
    rule="weekly",
)
NAIVE_SIMILAR_DAY = NaiveModel(
    name="naive_similar_day",
    version="1",
    description=(
        "Literature benchmark (Lago et al. 2021): Mon/Sat/Sun from one week before, "
        "Tue-Fri from the last known day."
    ),
    rule="similar_day",
)
