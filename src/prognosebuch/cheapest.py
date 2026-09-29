"""The cheapest 3-hour window of a delivery day, with probabilities.

For households on dynamic tariffs: when is the best time to run a flexible load? The
recommendation is the 3-hour window with the lowest mean median forecast. Its
probabilities come from scenarios: the median forecast plus each of the model's error
curves from the previous 60-90 days (the same curves its band is built from), aligned by
local wall-clock time. This keeps the shape of real forecast errors within a day.

- ``p_cheapest``: share of scenarios in which the recommended window is the cheapest one.
- ``p_near``: share of scenarios in which it is at most NEAR_EUR_MWH more expensive than
  the cheapest window (0.5 ct/kWh; practically as good).

Both are written into the forecast manifest before the auction and are scored later
against the real prices (hit rate, regret, Brier score), so the probabilities themselves
are testable. They are estimates, not promises.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from prognosebuch.timeutil import TZ, wall_clock

METHOD = "scenario-errors.v1"
WINDOW_HOURS = 3
WINDOW_SLOTS = WINDOW_HOURS * 4
NEAR_EUR_MWH = 5.0


def rolling_window_means(
    values: npt.NDArray[np.float64], w: int = WINDOW_SLOTS
) -> npt.NDArray[np.float64]:
    """Mean over each run of ``w`` consecutive quarter-hours along the last axis."""
    c = np.cumsum(np.concatenate([np.zeros((*values.shape[:-1], 1)), values], axis=-1), axis=-1)
    return (c[..., w:] - c[..., :-w]) / w


def scenario_matrix(point: pd.Series, errors: list[pd.Series]) -> npt.NDArray[np.float64]:
    """Rows = scenarios (point + one past error curve), columns = quarter-hours of the day."""
    target_clock = wall_clock(pd.DatetimeIndex(point.index))
    rows = []
    for err in errors:
        by_clock = err.groupby(wall_clock(pd.DatetimeIndex(err.index))).mean()
        aligned = by_clock.reindex(target_clock).ffill().bfill().fillna(0.0).to_numpy()
        rows.append(point.to_numpy(dtype=float) + aligned)
    return np.vstack(rows)


def _window(slots: pd.DatetimeIndex, j: int) -> dict[str, str]:
    start = slots[j]
    end = slots[j + WINDOW_SLOTS - 1] + pd.Timedelta(minutes=15)
    return {
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "start_local": start.tz_convert(TZ).isoformat(),
        "end_local": end.tz_convert(TZ).isoformat(),
    }


def cheapest_window(point: pd.Series, errors: list[pd.Series]) -> dict[str, Any]:
    slots = pd.DatetimeIndex(point.index)
    means = rolling_window_means(point.to_numpy(dtype=float))
    rec = int(np.argmin(means))
    scen = rolling_window_means(scenario_matrix(point, errors))
    best = scen.argmin(axis=1)
    best_val = scen.min(axis=1)
    counts = np.bincount(best, minlength=scen.shape[1])
    p = counts / len(errors)
    order = [int(j) for j in np.lexsort((means, -p))[:3]]
    return {
        **_window(slots, rec),
        "expected_mean_eur_mwh": round(float(means[rec]), 2),
        "day_mean_eur_mwh": round(float(point.mean()), 2),
        "p_cheapest": round(float(p[rec]), 3),
        "p_near": round(float(np.mean(scen[:, rec] - best_val <= NEAR_EUR_MWH)), 3),
        "n_scenarios": len(errors),
        "most_likely": [{**_window(slots, j), "p_cheapest": round(float(p[j]), 3)} for j in order],
    }


def realized(actual: pd.Series, start_utc: str) -> dict[str, Any]:
    """How the recommended window turned out against the real prices."""
    slots = pd.DatetimeIndex(actual.index)
    means = rolling_window_means(actual.to_numpy(dtype=float))
    j_rec = int(slots.get_indexer(pd.DatetimeIndex([pd.Timestamp(start_utc)]))[0])
    if j_rec < 0:
        raise ValueError(f"window start {start_utc} is not a quarter-hour of this day")
    j_best = int(np.argmin(means))
    regret = float(means[j_rec] - means[j_best])
    return {
        "actual_best_start_utc": slots[j_best].isoformat(),
        "actual_mean_recommended": float(means[j_rec]),
        "actual_mean_best": float(means[j_best]),
        "actual_day_mean": float(actual.mean()),
        "hit": j_rec == j_best,
        "near": regret <= NEAR_EUR_MWH,
        "regret_eur_mwh": regret,
    }
