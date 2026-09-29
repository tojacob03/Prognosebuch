from datetime import date

import pytest

from prognosebuch.timeutil import day_slots_utc, local_iso, wall_clock


@pytest.mark.parametrize(
    ("d", "n"),
    [
        (date(2026, 9, 30), 96),
        (date(2026, 10, 25), 100),  # autumn switch: 02:00-03:00 occurs twice
        (date(2027, 3, 28), 92),  # spring switch: 02:00-03:00 does not exist
        (date(2025, 10, 26), 100),
        (date(2026, 3, 29), 92),
    ],
)
def test_quarter_hours_per_day(d: date, n: int) -> None:
    assert len(day_slots_utc(d)) == n


def test_slots_are_contiguous_and_start_at_local_midnight() -> None:
    slots = day_slots_utc(date(2026, 10, 25))
    assert local_iso(slots[:1]) == ["2026-10-25T00:00:00+02:00"]
    assert local_iso(slots[-1:]) == ["2026-10-25T23:45:00+01:00"]
    assert (slots[1:] - slots[:-1]).unique().tolist() == [slots[1] - slots[0]]


def test_wall_clock_repeats_on_autumn_switch() -> None:
    labels = wall_clock(day_slots_utc(date(2026, 10, 25)))
    assert (labels == "02:15").sum() == 2
    assert (wall_clock(day_slots_utc(date(2027, 3, 28))) == "02:15").sum() == 0
