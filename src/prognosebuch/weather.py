"""Weather forecasts from Open-Meteo (CC BY 4.0), as they were known before the issue time.

Source: the Previous Runs API, which returns values as forecast a fixed lead time before
the valid time (``*_previous_dayN`` = N x 24 h earlier). Not the Historical Forecast API:
that one stitches together the first hours of each model run, which is close to observed
weather and would leak information into training.

Lead times: D+1 targets use ``previous_day2`` (48 h), D+2 targets ``previous_day3``
(72 h). A value is treated as available ``PUBLICATION_DELAY_H`` hours after the model run
it comes from, i.e. at ``valid - lead + delay``. With a 08:45 (Berlin) issue time every value
of the target day is available with several hours to spare; ``Inputs.known_at`` enforces it.

Only a handful of points are fetched and aggregated to national proxies: a wind power
proxy from onshore/offshore wind points, mean solar irradiance, and mean temperature.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prognosebuch.smard import JsonGetter, http_get_json
from prognosebuch.storage import read_parquet
from prognosebuch.timeutil import TIME_UNIT, local_date

API = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "icon_seamless"
ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"
LEADS = (2, 3)
PUBLICATION_DELAY_H = 6
FIRST_DAY = date(2024, 2, 19)
ARCHIVE_DIR = Path("inputs") / "weather" / "open_meteo_icon"


@dataclass(frozen=True)
class Point:
    name: str
    lat: float
    lon: float
    wind: bool
    solar: bool


POINTS: tuple[Point, ...] = (
    Point("German Bight (offshore)", 54.30, 7.00, wind=True, solar=False),
    Point("Husum", 54.48, 9.05, wind=True, solar=False),
    Point("Emden", 53.37, 7.21, wind=True, solar=False),
    Point("Rostock", 54.09, 12.10, wind=True, solar=False),
    Point("Uckermark", 53.10, 13.90, wind=True, solar=False),
    Point("Magdeburger Boerde", 52.10, 11.40, wind=True, solar=True),
    Point("Berlin", 52.52, 13.40, wind=False, solar=True),
    Point("Cologne", 50.94, 6.96, wind=False, solar=True),
    Point("Frankfurt", 50.11, 8.68, wind=False, solar=True),
    Point("Leipzig", 51.34, 12.37, wind=False, solar=True),
    Point("Stuttgart", 48.78, 9.18, wind=False, solar=True),
    Point("Munich", 48.14, 11.58, wind=False, solar=True),
)
VARIABLES = ("wind_speed_100m", "shortwave_radiation", "temperature_2m")
FEATURES = ("wind_power", "wind_speed", "solar", "temp")


def wind_power_proxy(v: np.ndarray) -> np.ndarray:
    """Normalised output of a generic turbine: 0 below 3 m/s, cubic to 12 m/s, flat above."""
    x = np.clip((v - 3.0) / 9.0, 0.0, 1.0)
    return np.asarray(np.where(v > 25.0, 0.0, x**3))


def _url(start: date, end: date) -> str:
    hourly = ",".join(f"{v}_previous_day{n}" for v in VARIABLES for n in LEADS)
    lat = ",".join(f"{p.lat}" for p in POINTS)
    lon = ",".join(f"{p.lon}" for p in POINTS)
    return (
        f"{API}?latitude={lat}&longitude={lon}&hourly={hourly}&models={MODEL}"
        f"&wind_speed_unit=ms&timezone=GMT&start_date={start}&end_date={end}"
    )


def _series(hourly: dict[str, Any], var: str, lead: int) -> np.ndarray:
    values = hourly[f"{var}_previous_day{lead}"]
    return np.array([np.nan if x is None else x for x in values], dtype=float)


def aggregate(payload: list[dict[str, Any]]) -> pd.DataFrame:
    """Open-Meteo multi-location response -> one row per valid hour and lead."""
    rows = []
    for n in LEADS:
        wind_v, solar_v, temp_v = [], [], []
        times = None
        for p, loc in zip(POINTS, payload, strict=True):
            h = loc["hourly"]
            times = h["time"]
            if p.wind:
                wind_v.append(_series(h, "wind_speed_100m", n))
            if p.solar:
                solar_v.append(_series(h, "shortwave_radiation", n))
            temp_v.append(_series(h, "temperature_2m", n))
        assert times is not None
        valid = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit(TIME_UNIT)
        wind = np.vstack(wind_v)
        # Radiation is the mean over the preceding hour: the value stamped hh+1 describes
        # the delivery hour starting at hh.
        solar = np.vstack(solar_v)
        solar = np.concatenate([solar[:, 1:], np.full((solar.shape[0], 1), np.nan)], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rows.append(
                pd.DataFrame(
                    {
                        "valid_utc": valid,
                        "lead_days": n,
                        "wind_power": np.nanmean(wind_power_proxy(wind), axis=0),
                        "wind_speed": np.nanmean(wind, axis=0),
                        "solar": np.nanmean(solar, axis=0),
                        "temp": np.nanmean(np.vstack(temp_v), axis=0),
                    }
                )
            )
    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=list(FEATURES), how="all")
    df["available_at_utc"] = df["valid_utc"] - pd.to_timedelta(
        df["lead_days"] * 24 - PUBLICATION_DELAY_H, unit="h"
    )
    return df.sort_values(["lead_days", "valid_utc"]).reset_index(drop=True)


def fetch(start: date, end: date, get_json: JsonGetter = http_get_json) -> pd.DataFrame:
    """Aggregated weather for valid days [start, end], fetched in chunks of 60 days."""
    parts = []
    d = start
    while d <= end:
        e = min(d + timedelta(days=59), end)
        # One extra day so the last hour's radiation (stamped at the next hour) is complete.
        chunk = aggregate(get_json(_url(d, e + timedelta(days=1))))
        limit = pd.Timestamp(e + timedelta(days=1), tz="UTC")
        parts.append(chunk[chunk["valid_utc"] < limit])
        d = e + timedelta(days=1)
    return pd.concat(parts, ignore_index=True) if parts else empty()


def empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "valid_utc": pd.DatetimeIndex([], tz="UTC").as_unit(TIME_UNIT),
            "lead_days": pd.Series([], dtype="int64"),
            **{f: pd.Series([], dtype="float64") for f in FEATURES},
            "available_at_utc": pd.DatetimeIndex([], tz="UTC").as_unit(TIME_UNIT),
        }
    )


def merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([old, new], ignore_index=True)
    df = df.drop_duplicates(["valid_utc", "lead_days"], keep="last")
    return df.sort_values(["lead_days", "valid_utc"]).reset_index(drop=True)


def _month_path(root: Path, month: str) -> Path:
    return root / ARCHIVE_DIR / f"{month}.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("valid_utc", "available_at_utc"):
        out[c] = pd.DatetimeIndex(out[c]).as_unit(TIME_UNIT)
    out["lead_days"] = out["lead_days"].astype("int64")
    for f in FEATURES:
        out[f] = out[f].astype(float).round(3)
    return out.sort_values(["lead_days", "valid_utc"]).reset_index(drop=True)


def load_weather(root: Path) -> pd.DataFrame:
    files = sorted((root / ARCHIVE_DIR).glob("*.parquet"))
    if not files:
        return empty()
    return _normalize(pd.concat([read_parquet(f) for f in files], ignore_index=True))


def save_weather(root: Path, df: pd.DataFrame) -> list[Path]:
    """Write one file per month of valid time; only files whose content changed are rewritten."""
    df = _normalize(df)
    months = pd.DatetimeIndex(df["valid_utc"]).strftime("%Y-%m")
    changed = []
    for month in sorted(set(months)):
        part = df[months == month].reset_index(drop=True)
        path = _month_path(root, month)
        if path.exists() and _normalize(read_parquet(path)).equals(part):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(path, index=False, compression="zstd")
        changed.append(path)
    return changed


def update_weather(
    root: Path, now_utc: datetime, get_json: JsonGetter = http_get_json, lookback_days: int = 14
) -> list[Path]:
    """Merge recent values (including already available ones for the next days)."""
    today = local_date(pd.Timestamp(now_utc))
    new = fetch(today - timedelta(days=lookback_days), today + timedelta(days=2), get_json)
    if new.empty:
        return []
    return save_weather(root, merge(load_weather(root), new))


def live_weather(
    root: Path, issue: date, get_json: JsonGetter = http_get_json
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Archive plus a fresh fetch around the issue day; returns (combined, fresh)."""
    fresh = fetch(issue - timedelta(days=10), issue + timedelta(days=2), get_json)
    return merge(load_weather(root), fresh), fresh
