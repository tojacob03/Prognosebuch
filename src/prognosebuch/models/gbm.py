"""Gradient boosting with quantile loss on weather forecasts, lagged prices and calendar.

Histogram-based gradient boosting (scikit-learn's ``HistGradientBoostingRegressor``, the
same algorithm family as LightGBM) with one model per quantile (10 %, 50 %, 90 %) and
horizon, pooled over the 24 delivery hours.

Features for delivery hour ``hh`` of target day ``t`` with horizon ``h`` (issue day t-h):

- weather forecast for that hour with lead ``h + 1`` days (Open-Meteo Previous Runs):
  wind power proxy, wind speed, solar irradiance, temperature; daily means of the target
  day and of the last known day, and their differences;
- prices: same hour on the last known day, one day earlier and one week before the target;
  mean, minimum and maximum of the last known day; mean of the last seven known days;
- calendar: hour, weekday, weekend, German public holiday, season (sine/cosine of day of
  year).

The target is the price minus the mean of the last seven known days, so the trees learn the
shape and weather effect while the level comes from recent prices (trees cannot
extrapolate beyond the price range seen in training).

Training: all target days from ``train_from`` up to the most recent Sunday on or before the
issue date (weekly re-estimation). Every training row only uses information that existed
at its own issue time, and the Sunday cutoff is before the issue date, so the fitted model
uses nothing from after the issue time either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from prognosebuch.models.base import QUANTILES, Inputs, InsufficientDataError
from prognosebuch.models.lear import LearModel, hourly_matrix, is_holiday
from prognosebuch.timeutil import TZ
from prognosebuch.weather import FEATURES as WEATHER_FEATURES

_CACHE: dict[tuple[Any, ...], list[HistGradientBoostingRegressor]] = {}
_CACHE_MAX = 64


def weather_matrix(weather: pd.DataFrame, lead: int, feature: str) -> pd.DataFrame:
    """Days x 24 local hours of one weather feature at one lead time."""
    w = weather[weather["lead_days"] == lead]
    if w.empty:
        return pd.DataFrame(columns=range(24), dtype=float)
    local = pd.DatetimeIndex(w["valid_utc"]).tz_convert(TZ)
    df = pd.DataFrame({"d": local.date, "h": local.hour, "v": w[feature].to_numpy(dtype=float)})
    m = df.groupby(["d", "h"])["v"].mean().unstack("h").reindex(columns=range(24))
    return m.ffill(axis=1).bfill(axis=1)


def sunday_on_or_before(d: date) -> date:
    return d - timedelta(days=(d.weekday() + 1) % 7)


@dataclass(frozen=True)
class GbmModel:
    name: str = "gbm"
    version: str = "1"
    description: str = (
        "Gradient boosting with quantile loss (P10/P50/P90) on Open-Meteo weather forecasts "
        "(lead h+1 days), lagged prices and calendar; re-estimated weekly."
    )
    train_from: date = date(2024, 3, 1)
    error_days: int = 60
    min_error_days: int = 20
    max_iter: int = 250
    learning_rate: float = 0.06
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    needs_weather: ClassVar[bool] = True

    # ------------------------------------------------------------ features

    def design(self, inputs: Inputs, h: int, until: date) -> tuple[pd.DataFrame, pd.Series]:
        """Long table (target day, hour) of features and the relative target."""
        if inputs.weather is None:
            raise InsufficientDataError(f"{self.name}: no weather data")
        daily = hourly_matrix(inputs.prices)
        if daily.empty:
            raise InsufficientDataError(f"{self.name}: no prices")
        days = pd.date_range(self.train_from - timedelta(days=10), until, freq="D").date
        p = daily.reindex(days)
        lead = h + 1
        wx = {f: weather_matrix(inputs.weather, lead, f).reindex(days) for f in WEATHER_FEATURES}

        def lag(m: pd.DataFrame, k: int) -> pd.DataFrame:
            return m.shift(k)

        last = lag(p, h)
        week_mean = p.mean(axis=1).rolling(7, min_periods=5).mean().shift(h)
        per_day = pd.DataFrame(
            {
                "p_last_mean": last.mean(axis=1),
                "p_last_min": last.min(axis=1),
                "p_last_max": last.max(axis=1),
                "p_week_mean": week_mean,
                "wind_power_day": wx["wind_power"].mean(axis=1),
                "solar_day": wx["solar"].mean(axis=1),
                "temp_day": wx["temp"].mean(axis=1),
                "wind_power_last_day": lag(wx["wind_power"], h).mean(axis=1),
                "solar_last_day": lag(wx["solar"], h).mean(axis=1),
            },
            index=p.index,
        )
        per_day["wind_power_change"] = per_day["wind_power_day"] - per_day["wind_power_last_day"]
        per_day["solar_change"] = per_day["solar_day"] - per_day["solar_last_day"]
        doy = np.array([t.timetuple().tm_yday for t in p.index], dtype=float)
        per_day["doy_sin"] = np.sin(2 * math.pi * doy / 365.25)
        per_day["doy_cos"] = np.cos(2 * math.pi * doy / 365.25)
        wd = np.array([t.weekday() for t in p.index])
        per_day["weekday"] = wd
        per_day["weekend"] = (wd >= 5).astype(float)
        per_day["holiday"] = np.array([is_holiday(t) for t in p.index], dtype=float)

        def stack(m: pd.DataFrame, name: str) -> pd.Series:
            """Days x hours -> Series indexed by (target, hour)."""
            idx = pd.MultiIndex.from_product([m.index, range(24)], names=["target", "hour"])
            return pd.Series(m.reindex(columns=range(24)).to_numpy().ravel(), index=idx, name=name)

        cols = [
            stack(last, "p_last_hour"),
            stack(lag(p, h + 1), "p_last2_hour"),
            stack(lag(p, 7), "p_week_hour"),
            *(stack(wx[f], f) for f in WEATHER_FEATURES),
        ]
        X = pd.concat(cols, axis=1)
        X = X.join(per_day, on="target")
        X["hour"] = X.index.get_level_values("hour").astype(float)
        y = stack(p, "price") - X["p_week_mean"]
        return X, y

    # ------------------------------------------------------------ fitting

    def _models(self, inputs: Inputs, h: int, anchor: date) -> list[HistGradientBoostingRegressor]:
        known = inputs.known_at(anchor)
        w = known.weather
        key = (
            self.name, self.version, h, anchor, len(known.prices),
            known.prices.index.max() if len(known.prices) else None,
            0 if w is None else len(w),
        )  # fmt: skip
        if key in _CACHE:
            return _CACHE[key]
        X, y = self.design(known, h, until=anchor)
        rows = (X.index.get_level_values("target") >= self.train_from) & y.notna()
        rows &= X[["p_last_hour", "p_week_hour", "p_week_mean"]].notna().all(axis=1)
        Xt, yt = X[rows], y[rows]
        if len(Xt) < 24 * 120:
            raise InsufficientDataError(f"{self.name}: only {len(Xt) // 24} training days")
        models = []
        for q in QUANTILES:
            m = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                categorical_features=None,
                random_state=0,
            )
            m.fit(Xt.to_numpy(dtype=float), yt.to_numpy(dtype=float))
            models.append(m)
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = models
        return models

    # ------------------------------------------------------------ prediction

    def quantiles_for(self, inputs: Inputs, issue: date, target: date) -> pd.DataFrame:
        """Quarter-hourly q10/q50/q90 for ``target`` using only inputs known on ``issue``."""
        h = (target - issue).days
        if h < 1:
            raise ValueError("target date must be after the issue date")
        known = inputs.known_at(issue)
        models = self._models(inputs, h, sunday_on_or_before(issue))
        X, _ = self.design(known, h, until=target)
        try:
            x = X.xs(target, level="target").sort_index()
        except KeyError as exc:
            raise InsufficientDataError(f"{self.name}: no features for {target}") from exc
        if x[["p_last_hour", "p_week_mean"]].isna().any().any():
            raise InsufficientDataError(f"{self.name}: missing lagged prices for {target}")
        if x[list(WEATHER_FEATURES)].isna().all().any():
            raise InsufficientDataError(f"{self.name}: missing weather forecast for {target}")
        base = x["p_week_mean"].to_numpy(dtype=float)
        preds = (
            np.column_stack([m.predict(x.to_numpy(dtype=float)) for m in models]) + base[:, None]
        )
        preds.sort(axis=1)  # quantile regression can cross; keep q10 <= q50 <= q90
        shape = LearModel.intra_hour_shape(known.prices, issue)
        cols = {
            name: LearModel.to_quarter_hours(preds[:, k], target, shape)
            for k, name in enumerate(("q10", "q50", "q90"))
        }
        return pd.DataFrame(cols)

    def point_for(self, inputs: Inputs, issue: date, target: date) -> pd.Series:
        return self.quantiles_for(inputs, issue, target)["q50"]

    def predict(self, info: Any, target_date: date) -> pd.DataFrame:
        return self.quantiles_for(info.inputs, info.issue_date, target_date)


GBM_V1 = GbmModel()
