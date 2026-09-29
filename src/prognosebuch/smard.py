"""Minimal client for the public SMARD chart-data API (Bundesnetzagentur | SMARD.de, CC BY 4.0).

The API serves each series in weekly chunks. ``index_<resolution>.json`` lists the chunk
start times (ms since epoch); each chunk is ``[[ms, value | null], ...]`` with the ms
marking the start of the interval in UTC. Values are overwritten in place, SMARD keeps
no vintages, so the time at which a value became available is not recoverable later.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd

BASE_URL = "https://www.smard.de/app/chart_data"
ATTRIBUTION = "Bundesnetzagentur | SMARD.de"


@dataclass(frozen=True)
class Series:
    filter_id: int
    region: str
    resolution: str
    description: str

    @property
    def key(self) -> str:
        return f"smard:{self.filter_id}:{self.region}:{self.resolution}"


PRICE_DE_LU = Series(4169, "DE-LU", "quarterhour", "Day-ahead price DE-LU (EUR/MWh)")

# Series whose publication times are logged by the availability probe.
PROBE_SERIES: tuple[Series, ...] = (
    PRICE_DE_LU,
    Series(411, "DE", "quarterhour", "Forecast: total load (MWh)"),
    Series(4362, "DE", "quarterhour", "Forecast: residual load (MWh)"),
    Series(122, "DE", "quarterhour", "Forecast generation: total (MWh)"),
    Series(123, "DE", "quarterhour", "Forecast generation: wind onshore (MWh)"),
    Series(3791, "DE", "quarterhour", "Forecast generation: wind offshore (MWh)"),
    Series(125, "DE", "quarterhour", "Forecast generation: photovoltaics (MWh)"),
    Series(5097, "DE", "quarterhour", "Forecast generation: wind and PV (MWh)"),
    Series(715, "DE", "quarterhour", "Forecast generation: other (MWh)"),
)

JsonGetter = Callable[[str], Any]


def http_get_json(url: str, retries: int = 4) -> Any:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = httpx.get(
                url,
                timeout=30.0,
                headers={"User-Agent": "prognosebuch (github.com/tojacob03/prognosebuch)"},
            )
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"SMARD request failed after {retries} attempts: {url}") from last


class SmardClient:
    def __init__(self, get_json: JsonGetter = http_get_json) -> None:
        self._get = get_json

    def chunk_starts(self, s: Series) -> list[int]:
        data = self._get(f"{BASE_URL}/{s.filter_id}/{s.region}/index_{s.resolution}.json")
        return sorted(int(t) for t in data["timestamps"])

    def chunk(self, s: Series, start_ms: int) -> pd.Series:
        name = f"{s.filter_id}_{s.region}_{s.resolution}_{start_ms}.json"
        data = self._get(f"{BASE_URL}/{s.filter_id}/{s.region}/{name}")
        rows = [(int(t), v) for t, v in data["series"] if v is not None]
        if not rows:
            return _empty()
        ms, vals = zip(*rows, strict=True)
        idx = pd.to_datetime(list(ms), unit="ms", utc=True)
        return pd.Series(
            vals, index=pd.DatetimeIndex(idx, name="delivery_start_utc"), dtype="float64"
        )

    def fetch(self, s: Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """All non-null values with start <= t < end, sorted by time."""
        starts = self.chunk_starts(s)
        start_ms, end_ms = int(start.value // 10**6), int(end.value // 10**6)
        wanted = [
            c
            for i, c in enumerate(starts)
            if c < end_ms and (i + 1 == len(starts) or starts[i + 1] > start_ms)
        ]
        parts = [self.chunk(s, c) for c in wanted]
        if not parts:
            return _empty()
        out = pd.concat(parts).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out[(out.index >= start) & (out.index < end)]

    def last_timestamp(self, s: Series) -> pd.Timestamp | None:
        """Start of the latest non-null interval of a series (checks the last two chunks)."""
        starts = self.chunk_starts(s)
        for c in reversed(starts[-2:]):
            ser = self.chunk(s, c)
            if len(ser):
                return pd.Timestamp(ser.index.max())
        return None


def _empty() -> pd.Series:
    return pd.Series(
        [], index=pd.DatetimeIndex([], tz="UTC", name="delivery_start_utc"), dtype="float64"
    )
