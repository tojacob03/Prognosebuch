"""Timetable for the clock workflow.

GitHub's cron scheduler turned out to be unreliable for this repository (runs up to six
hours late or dropped), which cost the first issue day. Dispatch events, on the other
hand, start within seconds, and a workflow may dispatch other workflows with its own
token. The clock workflow therefore waits for the next entry of this timetable, dispatches
the matching workflow, and hands over to a fresh copy of itself before the six-hour job
limit. The cron entries in the workflows stay as a backup.

All jobs are idempotent: a forecast run on a day that already has its forecasts, or an
evaluate run with nothing new to score, does nothing. Running a job twice is harmless;
missing the forecast window is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from prognosebuch.timeutil import TZ

UTC = ZoneInfo("UTC")

# Local (Europe/Berlin) times. The forecast window is 08:45-12:00; later entries are retries.
FORECAST_TIMES = (time(8, 55), time(9, 30), time(10, 30))
# Prices for D+1 appear around 12:45; SMARD can lag. Later entries are retries.
EVALUATE_TIMES = (time(14, 50), time(17, 40), time(21, 10))
# The availability probe runs every hour at this minute (UTC, so DST has no effect).
PROBE_MINUTE = 17


@dataclass(frozen=True, order=True)
class Event:
    at: datetime  # UTC
    workflow: str


def events_between(start: datetime, end: datetime) -> list[Event]:
    """All timetable entries with start < at <= end, sorted by time."""
    out: list[Event] = []
    day: date = start.astimezone(TZ).date() - timedelta(days=1)
    last_day = end.astimezone(TZ).date() + timedelta(days=1)
    while day <= last_day:
        for t in FORECAST_TIMES:
            out.append(Event(datetime.combine(day, t, tzinfo=TZ).astimezone(UTC), "forecast"))
        for t in EVALUATE_TIMES:
            out.append(Event(datetime.combine(day, t, tzinfo=TZ).astimezone(UTC), "evaluate"))
        day += timedelta(days=1)
    hour = start.astimezone(UTC).replace(minute=PROBE_MINUTE, second=0, microsecond=0)
    hour -= timedelta(hours=1)
    while hour <= end:
        out.append(Event(hour, "probe"))
        hour += timedelta(hours=1)
    return sorted(e for e in out if start < e.at <= end)


def next_event(after: datetime) -> Event:
    """The first timetable entry strictly after ``after``."""
    return events_between(after, after + timedelta(days=2))[0]
