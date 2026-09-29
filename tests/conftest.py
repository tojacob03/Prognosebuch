from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prognosebuch.smard import Series, SmardClient
from prognosebuch.timeutil import day_bounds_utc

FIXTURE = Path(__file__).parent / "fixtures" / "smard_prices_sample.csv"


class FakeClient(SmardClient):
    """Serves a fixed price series instead of calling SMARD."""

    def __init__(self, prices: pd.Series) -> None:
        self.prices = prices.sort_index()

    def fetch(self, s: Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        p = self.prices
        return p[(p.index >= start) & (p.index < end)]

    def last_timestamp(self, s: Series) -> pd.Timestamp | None:
        return pd.Timestamp(self.prices.index.max()) if len(self.prices) else None


def synthetic_prices(first: date, last: date, seed: int = 0) -> pd.Series:
    """Quarter-hourly prices with a daily shape, weekly pattern, noise and negative dips."""
    start, _ = day_bounds_utc(first)
    _, end = day_bounds_utc(last)
    idx = pd.date_range(start, end, freq="15min", inclusive="left", name="delivery_start_utc")
    local = idx.tz_convert("Europe/Berlin")
    hour = local.hour + local.minute / 60
    rng = np.random.default_rng(seed)
    base = 80 + 40 * np.sin((hour - 6) / 24 * 2 * np.pi) - 60 * np.exp(-((hour - 13) ** 2) / 6)
    weekend = np.where(local.weekday >= 5, -25.0, 0.0)
    values = base + weekend + rng.normal(0, 12, len(idx))
    return pd.Series(values, index=idx, dtype="float64")


def load_fixture() -> pd.Series:
    df = pd.read_csv(FIXTURE)
    idx = pd.DatetimeIndex(
        pd.to_datetime(df["delivery_start_utc"], utc=True), name="delivery_start_utc"
    )
    return pd.Series(df["price_eur_mwh"].to_numpy(dtype=float), index=idx)


@pytest.fixture
def real_prices() -> pd.Series:
    return load_fixture()
