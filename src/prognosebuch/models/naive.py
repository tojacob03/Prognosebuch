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

import numpy as np
import pandas as pd

from prognosebuch.models.base import InfoSet, InsufficientDataError, LeakError
from prognosebuch.timeutil import TZ, day_slots_utc, wall_clock

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
    window_days: int = 90
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

    def point(self, prices: pd.Series, issue_date: date, target_date: date) -> pd.Series:
        return profile_forecast(prices, self.source_day(issue_date, target_date), target_date)

    def error_quantiles(self, info: InfoSet, horizon: int) -> pd.DataFrame:
        """Empirical 10 %/90 % error quantiles of this rule by local hour.

        Uses only past (issue, target) pairs whose target day is fully known at the
        current issue date, i.e. target <= issue_date.
        """
        rows: list[pd.DataFrame] = []
        for k in range(self.window_days):
            past_target = info.issue_date - timedelta(days=k)
            past_issue = past_target - timedelta(days=horizon)
            try:
                fc = self.point(info.prices, past_issue, past_target)
            except InsufficientDataError:
                continue
            actual = info.prices.reindex(fc.index)
            err = (actual - fc).dropna()
            if len(err) < MIN_SOURCE_COVERAGE * len(fc):
                continue
            hours = pd.DatetimeIndex(err.index).tz_convert(TZ).hour
            rows.append(pd.DataFrame({"hour": hours, "err": err.to_numpy()}))
        if len(rows) < self.min_error_days:
            raise InsufficientDataError(
                f"{self.name}: only {len(rows)} days to estimate error quantiles "
                f"(need {self.min_error_days})"
            )
        errs = pd.concat(rows, ignore_index=True)
        g = errs.groupby("hour")["err"]
        q = pd.DataFrame({"e10": g.quantile(0.1), "e90": g.quantile(0.9)})
        return q.reindex(range(24)).ffill().bfill()

    def predict(self, info: InfoSet, target_date: date) -> pd.DataFrame:
        horizon = (target_date - info.issue_date).days
        if horizon < 1:
            raise ValueError("target date must be after the issue date")
        q50 = self.point(info.prices, info.issue_date, target_date)
        eq = self.error_quantiles(info, horizon)
        hours = pd.DatetimeIndex(q50.index).tz_convert(TZ).hour
        e10 = eq["e10"].reindex(hours).to_numpy()
        e90 = eq["e90"].reindex(hours).to_numpy()
        base = q50.to_numpy()
        out = pd.DataFrame(
            {
                "q10": np.minimum(base + e10, base),
                "q50": base,
                "q90": np.maximum(base + e90, base),
            },
            index=q50.index,
        )
        return out


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
