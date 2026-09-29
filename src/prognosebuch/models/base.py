"""Model interface and the information set a model is allowed to see."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd

from prognosebuch.timeutil import day_bounds_utc

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
QUANTILE_COLUMNS: tuple[str, ...] = ("q10", "q50", "q90")


class LeakError(RuntimeError):
    """Raised when data from after the information cutoff reaches a model."""


class InsufficientDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class InfoSet:
    """Everything a model may use for a forecast issued on ``issue_date`` (Europe/Berlin).

    Day-ahead prices are known for all delivery intervals up to the end of the issue day
    (they were published on the previous day). Nothing later may be present.
    """

    issue_date: date
    prices: pd.Series

    @property
    def prices_known_until_utc(self) -> pd.Timestamp:
        return day_bounds_utc(self.issue_date)[1]

    def __post_init__(self) -> None:
        if len(self.prices) and self.prices.index.max() >= self.prices_known_until_utc:
            raise LeakError(
                f"price at {self.prices.index.max()} is after the information cutoff "
                f"{self.prices_known_until_utc} for issue date {self.issue_date}"
            )

    @classmethod
    def cut(cls, issue_date: date, prices: pd.Series) -> InfoSet:
        """Build the information set by hard-filtering everything after the cutoff."""
        end = day_bounds_utc(issue_date)[1]
        return cls(issue_date=issue_date, prices=prices[prices.index < end])


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
