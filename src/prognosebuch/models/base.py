"""Model interface and the information set a model is allowed to see."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Protocol

import pandas as pd

from prognosebuch.timeutil import TZ, day_bounds_utc, normalize_index

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

# Forecasts are issued between 08:45 and 12:00 Europe/Berlin (see forecast.py).
EARLIEST_ISSUE = time(8, 45)
QUANTILE_COLUMNS: tuple[str, ...] = ("q10", "q50", "q90")


class LeakError(RuntimeError):
    """Raised when data from after the information cutoff reaches a model."""


class InsufficientDataError(RuntimeError):
    pass


def price_cutoff_utc(issue_date: date) -> pd.Timestamp:
    """Prices are known for all delivery intervals before the end of the issue day."""
    return day_bounds_utc(issue_date)[1]


def weather_cutoff_utc(issue_date: date) -> pd.Timestamp:
    """Weather values must have been available at the earliest issue time of the day."""
    local = datetime.combine(issue_date, EARLIEST_ISSUE, tzinfo=TZ)
    return pd.Timestamp(local).tz_convert("UTC")


def _cut_prices(prices: pd.Series, issue_date: date) -> pd.Series:
    prices = normalize_index(prices)
    return prices[prices.index < price_cutoff_utc(issue_date)]


def _cut_weather(weather: pd.DataFrame | None, issue_date: date) -> pd.DataFrame | None:
    if weather is None:
        return None
    return weather[weather["available_at_utc"] <= weather_cutoff_utc(issue_date)]


@dataclass(frozen=True)
class Inputs:
    """Raw model inputs, possibly covering more than one issue date (e.g. in a backtest).

    Models must call ``known_at(issue)`` before using them for a forecast issued on ``issue``.
    """

    prices: pd.Series
    weather: pd.DataFrame | None = None

    def known_at(self, issue_date: date) -> Inputs:
        return Inputs(
            prices=_cut_prices(self.prices, issue_date),
            weather=_cut_weather(self.weather, issue_date),
        )


@dataclass(frozen=True)
class InfoSet:
    """Everything a model may use for a forecast issued on ``issue_date`` (Europe/Berlin).

    - Day-ahead prices up to the end of the issue day (published the day before).
    - Weather forecasts that were available at the earliest issue time (08:45 local).
    Nothing later may be present; construction fails otherwise.
    """

    issue_date: date
    prices: pd.Series
    weather: pd.DataFrame | None = None

    @property
    def prices_known_until_utc(self) -> pd.Timestamp:
        return price_cutoff_utc(self.issue_date)

    @property
    def inputs(self) -> Inputs:
        return Inputs(self.prices, self.weather)

    def __post_init__(self) -> None:
        if len(self.prices) and self.prices.index.max() >= self.prices_known_until_utc:
            raise LeakError(
                f"price at {self.prices.index.max()} is after the information cutoff "
                f"{self.prices_known_until_utc} for issue date {self.issue_date}"
            )
        w = self.weather
        if w is not None and len(w):
            latest = w["available_at_utc"].max()
            if latest > weather_cutoff_utc(self.issue_date):
                raise LeakError(
                    f"weather value available at {latest} is after the cutoff "
                    f"{weather_cutoff_utc(self.issue_date)} for issue date {self.issue_date}"
                )

    @classmethod
    def cut(
        cls, issue_date: date, prices: pd.Series, weather: pd.DataFrame | None = None
    ) -> InfoSet:
        """Build the information set by hard-filtering everything after the cutoffs."""
        return cls(
            issue_date=issue_date,
            prices=_cut_prices(prices, issue_date),
            weather=_cut_weather(weather, issue_date),
        )


class Model(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def description(self) -> str: ...

    def predict(self, info: InfoSet, target_date: date) -> pd.DataFrame:
        """Quantile forecast for every quarter-hour of ``target_date``.

        Returns a frame indexed by ``delivery_start_utc`` with columns q10, q50, q90.
        """
        ...
