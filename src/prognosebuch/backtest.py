"""Rolling-origin backtest on past data. Strictly separate from the live book.

For every past issue date the models produce exactly what the live job would have
produced with prices known at that date (same ``point_for`` and the same band definition
from ``bands.py``). Point forecasts are computed once and reused for the bands, so the
backtest is equivalent to, but much cheaper than, calling ``predict`` for every day.

What a backtest cannot show: it is not pre-registered. Model design and settings were
chosen by someone who could see the past, so backtest numbers are optimistic by nature.
Only the live book (``forecasts/`` + ``scores/``) is a test of the future.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from prognosebuch.evaluate import summarize
from prognosebuch.metrics import mean_pinball
from prognosebuch.models.bands import PointModel, apply_bands, error_frame, hourly_error_quantiles
from prognosebuch.models.base import InsufficientDataError
from prognosebuch.registry import HORIZONS
from prognosebuch.storage import dump_json
from prognosebuch.timeutil import day_slots_utc, local_iso, normalize_index


@dataclass(frozen=True)
class BacktestResult:
    scores: pd.DataFrame
    daily: pd.DataFrame
    summary: dict[str, Any]


def _point(model: PointModel, prices: pd.Series, issue: date, target: date) -> pd.Series | None:
    try:
        return model.point_for(prices, issue, target)
    except InsufficientDataError:
        return None


def _days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=k) for k in range((last - first).days + 1)]


def backtest_model(
    model: PointModel,
    prices: pd.Series,
    first_target: date,
    last_target: date,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Score rows (same columns as ``scores/``) for all targets in [first, last]."""
    rows = []
    for h in HORIZONS:
        warm = first_target - timedelta(days=model.error_days + h)
        targets = _days(warm, last_target)
        points = Parallel(n_jobs=n_jobs)(
            delayed(_point)(model, prices, t - timedelta(days=h), t) for t in targets
        )
        by_target = {t: p for t, p in zip(targets, points, strict=True) if p is not None}
        err_by_target = {t: error_frame(prices, p) for t, p in by_target.items()}
        for t in _days(first_target, last_target):
            issue = t - timedelta(days=h)
            slots = day_slots_utc(t)
            y = prices.reindex(slots).to_numpy(dtype=float)
            status, q = "missed", np.full((len(slots), 3), np.nan)
            frames = [
                f
                for k in range(model.error_days)
                if (f := err_by_target.get(issue - timedelta(days=k))) is not None
            ]
            if t in by_target and len(frames) >= model.min_error_days:
                q = apply_bands(by_target[t], hourly_error_quantiles(frames)).to_numpy()
                status = "scored"
            err = y - q[:, 1]
            rows.append(
                pd.DataFrame(
                    {
                        "target_date": t,
                        "delivery_start_utc": slots,
                        "delivery_start_local": local_iso(slots),
                        "model": model.name,
                        "model_version": model.version,
                        "horizon_days": h,
                        "issue_date": issue,
                        "status": status,
                        "actual": y,
                        "q10": q[:, 0],
                        "q50": q[:, 1],
                        "q90": q[:, 2],
                        "error": err,
                        "abs_error": np.abs(err),
                        "sq_error": np.square(err),
                        "pinball": mean_pinball(y, q[:, 0], q[:, 1], q[:, 2]),
                        "in_band": (y >= q[:, 0]) & (y <= q[:, 2]),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def daily_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Per model, horizon and target day: MAE, RMSE, pinball, band coverage."""
    ok = scores[scores["status"] == "scored"]
    g = ok.groupby(["model", "model_version", "horizon_days", "target_date"])
    return g.agg(
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda s: float(np.sqrt(s.mean()))),
        pinball=("pinball", "mean"),
        coverage_80=("in_band", "mean"),
        n_slots=("abs_error", "size"),
    ).reset_index()


def run_backtest(
    models: Sequence[PointModel],
    prices: pd.Series,
    first_target: date,
    last_target: date,
    now_utc: datetime,
    n_jobs: int = 1,
) -> BacktestResult:
    prices = normalize_index(prices)
    parts = [backtest_model(m, prices, first_target, last_target, n_jobs) for m in models]
    scores = pd.concat(parts, ignore_index=True)
    summary = summarize(scores, now_utc)
    summary["kind"] = "backtest"
    summary["warning"] = (
        "Backtest on past data, not pre-registered. Model choices were made with knowledge "
        "of the past; treat these numbers as optimistic. The live book is the real test."
    )
    summary["period"] = {
        "first_target": first_target.isoformat(),
        "last_target": last_target.isoformat(),
    }
    return BacktestResult(scores, daily_table(scores), summary)


def write_backtest(res: BacktestResult, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_bytes(dump_json(res.summary))
    daily_path = out_dir / "daily.csv"
    res.daily.to_csv(daily_path, index=False, float_format="%.4f")
    return [summary_path, daily_path]
