"""Time handling for the DE-LU day-ahead market.

All stored timestamps are UTC. A "delivery day" is a calendar day in Europe/Berlin,
which has 96 quarter-hours normally, 92 on the spring DST switch and 100 on the
autumn DST switch.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

TZ = ZoneInfo("Europe/Berlin")
SLOT_FREQ = "15min"


def local_midnight_utc(d: date) -> pd.Timestamp:
    """UTC instant of 00:00 Europe/Berlin on day ``d`` (midnight always exists in Berlin)."""
    return pd.Timestamp(datetime.combine(d, time(0), tzinfo=TZ)).tz_convert("UTC")


def day_bounds_utc(d: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Half-open UTC interval [start, end) covering local delivery day ``d``."""
    return local_midnight_utc(d), local_midnight_utc(d + timedelta(days=1))


def day_slots_utc(d: date) -> pd.DatetimeIndex:
    """Start instants (UTC) of all quarter-hours of local delivery day ``d``."""
    start, end = day_bounds_utc(d)
    return pd.date_range(start, end, freq=SLOT_FREQ, inclusive="left", name="delivery_start_utc")


def to_local(ts: pd.Timestamp | datetime) -> datetime:
    """Convert an aware timestamp to a Europe/Berlin datetime."""
    return pd.Timestamp(ts).tz_convert(TZ).to_pydatetime()


def local_date(ts: pd.Timestamp | datetime) -> date:
    return to_local(ts).date()


def local_iso(idx: pd.DatetimeIndex) -> list[str]:
    """ISO 8601 strings with explicit UTC offset, e.g. ``2026-10-25T02:00:00+02:00``."""
    return [t.isoformat() for t in idx.tz_convert(TZ)]


def wall_clock(idx: pd.DatetimeIndex) -> pd.Index:
    """Local wall-clock label ``HH:MM`` for each instant (ambiguous on the autumn switch)."""
    return pd.Index(idx.tz_convert(TZ).strftime("%H:%M"))


def utc_now() -> datetime:
    return datetime.now(tz=ZoneInfo("UTC"))
