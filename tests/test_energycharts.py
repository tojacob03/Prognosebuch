import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prognosebuch import energycharts
from prognosebuch.forecast import RunInfo, run_forecast
from prognosebuch.history import FALLBACK_LOG, load_actuals, update_actuals
from prognosebuch.storage import forecast_paths
from prognosebuch.timeutil import day_bounds_utc, day_slots_utc
from tests.conftest import FakeClient, fast_live_models, seeded_client, synthetic_prices

ISSUE = date(2026, 10, 10)


def fake_getter(prices: pd.Series) -> Any:
    def get(url: str) -> dict[str, Any]:
        q = dict(part.split("=") for part in url.split("?", 1)[1].split("&"))
        first, last = date.fromisoformat(q["start"]), date.fromisoformat(q["end"])
        lo, hi = day_slots_utc(first)[0], day_slots_utc(last)[-1]
        p = prices[(prices.index >= lo) & (prices.index <= hi)]
        return {
            "license_info": "CC BY 4.0 from Bundesnetzagentur | SMARD.de",
            "unix_seconds": [int(t.timestamp()) for t in p.index],
            "price": p.to_numpy().tolist(),
        }

    return get


def with_gap(prices: pd.Series, day: date) -> pd.Series:
    return prices.drop(day_slots_utc(day), errors="ignore")


def test_fill_gaps_only_fills_missing_and_never_overrides() -> None:
    full = synthetic_prices(date(2026, 10, 1), date(2026, 10, 12), seed=1)
    smard = with_gap(full, date(2026, 10, 10))
    smard = smard.mask(smard.index == day_slots_utc(date(2026, 10, 9))[0], 123.0)  # differs
    out, gap = energycharts.fill_gaps(
        smard, date(2026, 10, 1), date(2026, 10, 11), fake_getter(full)
    )
    assert gap.days == ["2026-10-10"] and len(gap.filled) == 96 and gap.error is None
    assert out.reindex(day_slots_utc(date(2026, 10, 10))).notna().all()
    assert out.loc[day_slots_utc(date(2026, 10, 9))[0]] == 123.0
    assert np.allclose(
        out.reindex(full.index).to_numpy(),
        np.where(full.index == day_slots_utc(date(2026, 10, 9))[0], 123.0, full.to_numpy()),
    )


def test_fill_gaps_without_gaps_or_with_errors_is_harmless() -> None:
    full = synthetic_prices(date(2026, 10, 1), date(2026, 10, 5), seed=2)
    out, gap = energycharts.fill_gaps(full, date(2026, 10, 1), date(2026, 10, 5), fake_getter(full))
    assert out is full and gap.filled.empty and gap.error is None

    def broken(url: str) -> Any:
        raise RuntimeError("down")

    out, gap = energycharts.fill_gaps(
        with_gap(full, date(2026, 10, 3)), date(2026, 10, 1), date(2026, 10, 5), broken
    )
    assert (
        gap.error
        and "down" in gap.error
        and out.reindex(day_slots_utc(date(2026, 10, 3))).isna().all()
    )


def test_forecast_survives_a_missing_issue_day_on_smard(tmp_path: Path) -> None:
    full = synthetic_prices(date(2026, 1, 1), ISSUE, seed=3)
    known = full[full.index < day_bounds_utc(ISSUE)[1]]
    smard = with_gap(known, ISSUE)  # exactly what happened on 2026-10-10
    models = fast_live_models(live_since=ISSUE)
    getter = fake_getter(known)
    res = run_forecast(
        tmp_path,
        datetime(2026, 10, 10, 6, 55, tzinfo=UTC),
        seeded_client(tmp_path, smard),
        RunInfo(None, None),
        models,
        price_fallback=lambda p, a, b: energycharts.fill_gaps(p, a, b, getter),
    )
    assert not res.failed and len(res.written) == 2 * len(models)
    m = json.loads(forecast_paths(tmp_path, ISSUE, "naive_last_day", "1")[1].read_text())
    fb = m["data_cutoffs"][energycharts.KEY]
    assert fb["filled_days_local"] == ["2026-10-10"] and fb["filled_values"] == 96


def test_update_actuals_fills_and_logs(tmp_path: Path) -> None:
    full = synthetic_prices(date(2026, 9, 15), date(2026, 10, 11), seed=4)
    smard = with_gap(full, date(2026, 10, 10))
    now = datetime(2026, 10, 10, 13, 0, tzinfo=UTC)
    update_actuals(tmp_path, FakeClient(smard), now, fallback_get_json=fake_getter(full))
    a = load_actuals(tmp_path)
    assert a.reindex(day_slots_utc(date(2026, 10, 10))).notna().all()
    log = pd.read_csv(tmp_path / FALLBACK_LOG)
    assert log["delivery_date_local"].tolist() == ["2026-10-10"]
    assert log["quarter_hours_filled"].tolist() == [96]
    # a second run has nothing left to fill and does not log again
    update_actuals(
        tmp_path, FakeClient(smard), now + timedelta(hours=3), fallback_get_json=fake_getter(full)
    )
    assert len(pd.read_csv(tmp_path / FALLBACK_LOG)) == 1
