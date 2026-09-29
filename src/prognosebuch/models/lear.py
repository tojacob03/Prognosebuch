"""LEAR: Lasso-Estimated AutoRegressive model.

Independent implementation after J. Lago, G. Marcjasz, B. De Schutter, R. Weron (2021),
"Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best
practices and an open-access benchmark", Applied Energy 293, 116983.

Adaptation to this project:
- One linear model per delivery hour and horizon, on hourly means of the price series.
- Inputs: all 24 hourly prices of the days t-h, t-h-1, t-h-2 and t-7 (for D+1: t-1, t-2,
  t-3, t-7 as in the paper; for D+2 shifted by one day, because t-1 is not yet known),
  day-of-week dummies and a German public holiday flag for the target day. No exogenous
  load or renewable forecasts yet (not reliably available at issue time, see
  DATA_SOURCES.md).
- Variance-stabilising transform: asinh of median/MAD-normalised values.
- L1 penalty chosen per hour by the Akaike information criterion on the LARS path.
- Average over several calibration windows (days).
- Quarter-hours: the hourly forecast plus the average intra-hour shape of the last 28 days.
- 80 % band: empirical error quantiles (per local hour) of the same model's point
  forecasts for the previous ``error_days`` target days, each recomputed as it would have
  been issued then (see ``bands.py``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache

import holidays
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoLarsIC

from prognosebuch.models.bands import predict_with_bands
from prognosebuch.models.base import InfoSet, InsufficientDataError
from prognosebuch.timeutil import TZ, day_bounds_utc, day_slots_utc

MIN_HOURS_PER_DAY = 22
SHAPE_DAYS = 28
QH_AUCTION_START = date(2025, 10, 1)


@lru_cache(maxsize=16)
def _german_holidays(year: int) -> frozenset[date]:
    return frozenset(holidays.country_holidays("DE", years=[year]).keys())


def is_holiday(d: date) -> bool:
    return d in _german_holidays(d.year)


def hourly_matrix(prices: pd.Series) -> pd.DataFrame:
    """Days (Europe/Berlin) x 24 local hours of mean prices, on a complete daily index.

    A doubled hour (autumn switch) is averaged; a missing hour (spring switch) or a small
    gap is filled from the previous hour. Days with fewer than 22 hours are left empty.
    """
    if prices.empty:
        return pd.DataFrame(columns=range(24), dtype=float)
    local = pd.DatetimeIndex(prices.index).tz_convert(TZ)
    df = pd.DataFrame({"d": local.date, "h": local.hour, "p": prices.to_numpy(dtype=float)})
    m = df.groupby(["d", "h"])["p"].mean().unstack("h").reindex(columns=range(24))
    n_hours = m.notna().sum(axis=1)
    m = m.ffill(axis=1).bfill(axis=1)
    m[n_hours < MIN_HOURS_PER_DAY] = np.nan
    full = pd.date_range(min(m.index), max(m.index), freq="D").date
    return m.reindex(full)


def _lag_days(h: int) -> tuple[int, ...]:
    return (h, h + 1, h + 2, 7)


def design(daily: pd.DataFrame, h: int, until: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature matrix X (index = target day) and targets Y (24 hours) for horizon ``h``.

    Rows cover target days up to ``until``; features of row t only use days <= t - h.
    """
    idx = pd.date_range(min(daily.index), until, freq="D").date
    d = daily.reindex(idx)
    blocks = []
    for lag in _lag_days(h):
        b = d.shift(lag)
        b.columns = pd.Index([f"lag{lag}_h{c:02d}" for c in range(24)])
        blocks.append(b)
    cal = pd.DataFrame(index=d.index)
    wd = np.array([t.weekday() for t in d.index])
    for k in range(6):  # Monday..Saturday; Sunday is the baseline
        cal[f"dow{k}"] = (wd == k).astype(float)
    cal["holiday"] = np.array([is_holiday(t) for t in d.index], dtype=float)
    X = pd.concat([*blocks, cal], axis=1)
    return X, d


def _scale(
    train: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    med = np.median(train, axis=0)
    mad = np.median(np.abs(train - med), axis=0) / 0.6745
    mad = np.where(mad > 1e-9, mad, 1.0)
    return med, mad


@dataclass(frozen=True)
class LearModel:
    name: str = "lear"
    version: str = "1"
    description: str = (
        "LEAR (Lago et al. 2021): per-hour lasso autoregression on prices of t-h..t-h-2 and "
        "t-7, weekday and holiday dummies; asinh transform; windows 182/364/728 days."
    )
    windows: tuple[int, ...] = (182, 364, 728)
    error_days: int = 60
    min_error_days: int = 20
    n_price_cols: int = field(default=96, repr=False)

    # ---------------------------------------------------------------- point forecast

    def hourly_point(
        self, X: pd.DataFrame, Y: pd.DataFrame, issue: date, h: int
    ) -> npt.NDArray[np.float64] | None:
        """24 hourly point forecasts for target issue + h, or None if data is missing."""
        target = issue + timedelta(days=h)
        if target not in X.index:
            return None
        x_new = X.loc[[target]].to_numpy(dtype=float)
        if np.isnan(x_new).any():
            return None
        preds = []
        for w in self.windows:
            rows = pd.date_range(issue - timedelta(days=w - 1), issue, freq="D").date
            Xw = X.reindex(rows).to_numpy(dtype=float)
            Yw = Y.reindex(rows).to_numpy(dtype=float)
            ok = ~(np.isnan(Xw).any(axis=1) | np.isnan(Yw).any(axis=1))
            if ok.sum() < 0.9 * w or ok.sum() <= Xw.shape[1] + 2:
                continue
            preds.append(self._fit_predict(Xw[ok], Yw[ok], x_new))
        if not preds:
            return None
        return np.mean(preds, axis=0)

    def _fit_predict(
        self,
        Xw: npt.NDArray[np.float64],
        Yw: npt.NDArray[np.float64],
        x_new: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        p = self.n_price_cols
        xm, xs = _scale(Xw[:, :p])
        ym, ys = _scale(Yw)
        Xt = np.hstack([np.arcsinh((Xw[:, :p] - xm) / xs), Xw[:, p:]])
        xn = np.hstack([np.arcsinh((x_new[:, :p] - xm) / xs), x_new[:, p:]])
        Yt = np.arcsinh((Yw - ym) / ys)
        out = np.empty(24)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for hh in range(24):
                model = LassoLarsIC(criterion="aic", max_iter=2500)
                model.fit(Xt, Yt[:, hh])
                out[hh] = model.predict(xn)[0]
        return np.sinh(out) * ys + ym

    # ---------------------------------------------------------------- quarter-hours

    @staticmethod
    def intra_hour_shape(prices: pd.Series, issue: date) -> pd.Series:
        """Mean deviation of each quarter-hour from its hourly mean, by local HH:MM."""
        start = max(issue - timedelta(days=SHAPE_DAYS - 1), QH_AUCTION_START)
        idx = pd.DatetimeIndex(prices.index)
        lo = day_slots_utc(start)[0]
        hi = day_slots_utc(issue)[-1]
        p = prices[(idx >= lo) & (idx <= hi)]
        if p.empty:
            return pd.Series(dtype=float)
        local = pd.DatetimeIndex(p.index).tz_convert(TZ)
        df = pd.DataFrame(
            {"d": local.date, "h": local.hour, "clock": local.strftime("%H:%M"), "p": p.to_numpy()}
        )
        df["dev"] = df["p"] - df.groupby(["d", "h"])["p"].transform("mean")
        return df.groupby("clock")["dev"].mean()

    @staticmethod
    def to_quarter_hours(
        hourly: npt.NDArray[np.float64], target: date, shape: pd.Series
    ) -> pd.Series:
        slots = day_slots_utc(target)
        local = slots.tz_convert(TZ)
        base = hourly[np.asarray(local.hour)]
        adj = shape.reindex(local.strftime("%H:%M")).fillna(0.0).to_numpy()
        return pd.Series(base + adj, index=slots)

    # ---------------------------------------------------------------- Model protocol

    def point_for(self, prices: pd.Series, issue: date, target: date) -> pd.Series:
        h = (target - issue).days
        known = prices[pd.DatetimeIndex(prices.index) < day_bounds_utc(issue)[1]]
        X, Y = design(hourly_matrix(known), h, until=target)
        hourly = self.hourly_point(X, Y, issue, h)
        if hourly is None:
            raise InsufficientDataError(f"{self.name}: not enough history for {target}")
        return self.to_quarter_hours(hourly, target, self.intra_hour_shape(known, issue))

    def predict(self, info: InfoSet, target_date: date) -> pd.DataFrame:
        return predict_with_bands(self, info, target_date)


LEAR_V1 = LearModel()
