from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prognosebuch.history import merge_into_archive
from prognosebuch.models.lear import LearModel
from prognosebuch.registry import LIVE_MODELS, LiveModel
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
    return pd.Series(values, index=idx.as_unit("ns"), dtype="float64")


def load_fixture() -> pd.Series:
    df = pd.read_csv(FIXTURE)
    idx = pd.DatetimeIndex(
        pd.to_datetime(df["delivery_start_utc"], utc=True), name="delivery_start_utc"
    )
    return pd.Series(df["price_eur_mwh"].to_numpy(dtype=float), index=idx.as_unit("ns"))


@pytest.fixture
def real_prices() -> pd.Series:
    return load_fixture()


def fast_live_models(live_since: date | None = None) -> tuple[LiveModel, ...]:
    """The registered live models, with LEAR shrunk to short windows so tests stay fast.

    Same classes and code paths; only calibration windows and error days are smaller.
    """
    out = []
    for lm in LIVE_MODELS:
        model = lm.model
        if isinstance(model, LearModel):
            model = replace(model, windows=(120,), error_days=6, min_error_days=4)
        out.append(LiveModel(model, live_since=live_since or lm.live_since))
    return tuple(out)


def seeded_client(root: Path, prices: pd.Series) -> FakeClient:
    """Write ``prices`` into the actuals archive under ``root`` and serve them as SMARD."""
    merge_into_archive(root, prices)
    return FakeClient(prices)
