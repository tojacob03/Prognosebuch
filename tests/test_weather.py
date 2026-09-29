from datetime import date
from pathlib import Path

import numpy as np

from prognosebuch.weather import (
    load_weather,
    merge,
    save_weather,
    wind_power_proxy,
)
from tests.conftest import synthetic_weather


def test_monthly_archive_rewrites_only_changed_months(tmp_path: Path) -> None:
    w = synthetic_weather(date(2026, 7, 1), date(2026, 9, 30), seed=1)
    first = save_weather(tmp_path, w)
    assert [p.name for p in first] == [
        "2026-06.parquet",
        "2026-07.parquet",
        "2026-08.parquet",
        "2026-09.parquet",
    ]
    assert len(load_weather(tmp_path)) == len(w)
    newer = w[w["valid_utc"] >= w["valid_utc"].max() - np.timedelta64(3, "D")].copy()
    newer["temp"] += 1.0
    changed = save_weather(tmp_path, merge(load_weather(tmp_path), newer))
    assert [p.name for p in changed] == ["2026-09.parquet"]
    assert save_weather(tmp_path, load_weather(tmp_path)) == []


def test_wind_power_proxy() -> None:
    v = np.array([0.0, 3.0, 7.5, 12.0, 20.0, 26.0])
    assert np.allclose(wind_power_proxy(v), [0, 0, 0.125, 1, 1, 0])
