"""Fallback source for day-ahead prices: the Energy-Charts API of Fraunhofer ISE.

Energy-Charts republishes the same SMARD price series (license info returned by the API:
"CC BY 4.0 from Bundesnetzagentur | SMARD.de"). On 2026-10-10 SMARD's own API lacked a
whole delivery day while Energy-Charts had it; on the 3,840 quarter-hours both had in
September/October 2026 the values were identical. Energy-Charts is therefore only used to
fill quarter-hours that are missing on SMARD, never to override SMARD values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from prognosebuch.smard import JsonGetter, http_get_json
from prognosebuch.timeutil import TIME_UNIT, TZ, day_slots_utc

API = "https://api.energy-charts.info/price"
KEY = "energy-charts:price:DE-LU"
ATTRIBUTION = "Bundesnetzagentur | SMARD.de, via Energy-Charts (Fraunhofer ISE)"


def fetch(first: date, last: date, get_json: JsonGetter = http_get_json) -> pd.Series:
    """DE-LU day-ahead prices for local delivery days first..last (inclusive)."""
    data = get_json(f"{API}?bzn=DE-LU&start={first}&end={last}")
    idx = pd.DatetimeIndex(pd.to_datetime(data["unix_seconds"], unit="s", utc=True))
    s = pd.Series(data["price"], index=idx.as_unit(TIME_UNIT), dtype="float64").dropna()
    lo, hi = day_slots_utc(first)[0], day_slots_utc(last)[-1]
    return s[(s.index >= lo) & (s.index <= hi)].rename_axis("delivery_start_utc")


@dataclass
class GapFill:
    """What was filled from Energy-Charts, for manifests and logs."""

    filled: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    error: str | None = None

    @property
    def days(self) -> list[str]:
        if self.filled.empty:
            return []
        local = pd.DatetimeIndex(self.filled.index).tz_convert(TZ)
        return sorted({d.isoformat() for d in local.date})


def incomplete_days(prices: pd.Series, first: date, last: date) -> list[date]:
    out = []
    d = first
    while d <= last:
        if prices.reindex(day_slots_utc(d)).isna().any():
            out.append(d)
        d += timedelta(days=1)
    return out


def fill_gaps(
    prices: pd.Series, first: date, last: date, get_json: JsonGetter = http_get_json
) -> tuple[pd.Series, GapFill]:
    """Fill quarter-hours missing in ``prices`` on days first..last from Energy-Charts."""
    days = incomplete_days(prices, first, last)
    if not days:
        return prices, GapFill()
    try:
        ec = fetch(days[0], days[-1], get_json)
    except Exception as exc:  # the fallback must never break the main path
        return prices, GapFill(error=f"{type(exc).__name__}: {exc}")
    wanted = pd.DatetimeIndex([t for d in days for t in day_slots_utc(d)])
    missing = wanted[prices.reindex(wanted).isna().to_numpy()]
    filled = ec.reindex(missing).dropna()
    if filled.empty:
        return prices, GapFill()
    out = pd.concat([prices.drop(filled.index, errors="ignore"), filled]).sort_index()
    return out, GapFill(filled=filled)
