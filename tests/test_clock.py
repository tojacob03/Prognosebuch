from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from prognosebuch.cli import main
from prognosebuch.clock import events_between, next_event
from prognosebuch.forecast import DEADLINE
from prognosebuch.models.base import EARLIEST_ISSUE

BERLIN = ZoneInfo("Europe/Berlin")
UTC = ZoneInfo("UTC")


def local(y: int, m: int, d: int, hh: int, mm: int) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=BERLIN)


def test_forecast_entries_fall_inside_the_issue_window_all_year() -> None:
    start = local(2026, 10, 1, 0, 0)
    events = events_between(start, start + timedelta(days=400))
    forecasts = [e for e in events if e.workflow == "forecast"]
    assert len(forecasts) == 3 * 400
    for e in forecasts:
        t = e.at.astimezone(BERLIN).time()
        assert EARLIEST_ISSUE <= t < DEADLINE


def test_next_event_in_the_morning_is_the_forecast() -> None:
    ev = next_event(local(2026, 10, 1, 8, 20))
    assert ev.workflow == "forecast"
    assert ev.at.astimezone(BERLIN) == local(2026, 10, 1, 8, 55)


def test_winter_and_summer_time() -> None:
    summer = next_event(local(2026, 10, 24, 8, 40))  # CEST
    winter = next_event(local(2026, 10, 26, 8, 40))  # CET
    assert summer.at.astimezone(UTC).hour == 6 and winter.at.astimezone(UTC).hour == 7
    assert summer.at.astimezone(BERLIN).strftime("%H:%M") == "08:55"
    assert winter.at.astimezone(BERLIN).strftime("%H:%M") == "08:55"


def test_probe_every_hour_and_strictly_after() -> None:
    t0 = datetime(2026, 10, 1, 3, 17, tzinfo=UTC)
    ev = next_event(t0)
    assert ev.workflow == "probe" and ev.at == t0 + timedelta(hours=1)
    day = events_between(t0, t0 + timedelta(days=1))
    assert sum(e.workflow == "probe" for e in day) == 24
    assert sum(e.workflow == "evaluate" for e in day) == 3


def test_cli_prints_workflow_and_epoch(capsys) -> None:  # type: ignore[no-untyped-def]
    after = int(local(2026, 10, 1, 14, 30).timestamp())
    assert main(["clock-next", "--after", str(after)]) == 0
    wf, epoch = capsys.readouterr().out.split()
    assert wf == "evaluate"
    assert datetime.fromtimestamp(int(epoch), tz=BERLIN) == local(2026, 10, 1, 14, 50)
