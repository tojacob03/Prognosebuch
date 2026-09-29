"""Empirical 80 % bands from a model's own recent out-of-sample errors.

Shared by live forecasts and the backtest so that both use exactly the same definition:
errors (actual - point forecast) of the previous N target days that were already known at
the issue date, pooled per local hour; q10/q90 = point + 10 %/90 % error quantile, clipped
so that q10 <= q50 <= q90.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from prognosebuch.models.base import InfoSet, InsufficientDataError
from prognosebuch.timeutil import TZ

MIN_COVERAGE = 0.9


def error_frame(actual: pd.Series, point: pd.Series) -> pd.DataFrame | None:
    """Errors of one target day by local hour, or None if too many actuals are missing."""
    err = (actual.reindex(point.index) - point).dropna()
    if len(err) < MIN_COVERAGE * len(point):
        return None
    return hour_frame(err)


def hourly_error_quantiles(frames: list[pd.DataFrame]) -> pd.DataFrame:
    errs = pd.concat(frames, ignore_index=True)
    g = errs.groupby("hour")["err"]
    q = pd.DataFrame({"e10": g.quantile(0.1), "e90": g.quantile(0.9)})
    return q.reindex(range(24)).ffill().bfill()


def apply_bands(point: pd.Series, eq: pd.DataFrame) -> pd.DataFrame:
    hours = pd.DatetimeIndex(point.index).tz_convert(TZ).hour
    base = point.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "q10": np.minimum(base + eq["e10"].reindex(hours).to_numpy(), base),
            "q50": base,
            "q90": np.maximum(base + eq["e90"].reindex(hours).to_numpy(), base),
        },
        index=point.index,
    )


class PointModel(Protocol):
    """A model defined by a point forecast rule; bands come from its own past errors."""

    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def error_days(self) -> int: ...
    @property
    def min_error_days(self) -> int: ...

    def point_for(self, prices: pd.Series, issue: date, target: date) -> pd.Series:
        """Quarter-hourly point forecast for ``target`` using only prices known on ``issue``."""
        ...

    def predict(self, info: InfoSet, target_date: date) -> pd.DataFrame: ...


def past_errors(model: PointModel, prices: pd.Series, issue: date, horizon: int) -> list[pd.Series]:
    """Errors (actual - point) of the model for the previous target days known at ``issue``.

    Each past forecast is recomputed exactly as it would have been issued on its own issue
    date. Days with more than 10 % missing actuals are skipped.
    """
    out = []
    for k in range(model.error_days):
        past_target = issue - timedelta(days=k)
        try:
            fc = model.point_for(prices, past_target - timedelta(days=horizon), past_target)
        except InsufficientDataError:
            continue
        err = (prices.reindex(fc.index) - fc).dropna()
        if len(err) >= MIN_COVERAGE * len(fc):
            out.append(err)
    return out


def hour_frame(err: pd.Series) -> pd.DataFrame:
    hours = pd.DatetimeIndex(err.index).tz_convert(TZ).hour
    return pd.DataFrame({"hour": hours, "err": err.to_numpy()})


def predict_with_errors(
    model: PointModel, info: InfoSet, target: date
) -> tuple[pd.DataFrame, list[pd.Series]]:
    """Quantile forecast plus the past error curves the band was estimated from."""
    horizon = (target - info.issue_date).days
    if horizon < 1:
        raise ValueError("target date must be after the issue date")
    point = model.point_for(info.prices, info.issue_date, target)
    errors = past_errors(model, info.prices, info.issue_date, horizon)
    if len(errors) < model.min_error_days:
        raise InsufficientDataError(
            f"{model.name}: only {len(errors)} days to estimate error quantiles "
            f"(need {model.min_error_days})"
        )
    frames = [hour_frame(e) for e in errors]
    return apply_bands(point, hourly_error_quantiles(frames)), errors


def predict_with_bands(model: PointModel, info: InfoSet, target: date) -> pd.DataFrame:
    return predict_with_errors(model, info, target)[0]
