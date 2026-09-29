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
from prognosebuch.models.base import Inputs, InsufficientDataError
from prognosebuch.registry import HORIZONS
from prognosebuch.storage import dump_json
from prognosebuch.timeutil import day_slots_utc, local_iso, normalize_index


@dataclass(frozen=True)
class BacktestResult:
    scores: pd.DataFrame
    daily: pd.DataFrame
    summary: dict[str, Any]


def _point(
    model: PointModel, inputs: Inputs, issue: date, target: date
) -> pd.Series | pd.DataFrame | None:
    """Point forecast, or the full quantile frame for models with native quantiles."""
    try:
        raw = getattr(model, "raw_quantiles_for", None)
        if raw is not None:  # calibrated in the main process, see backtest_model
            return raw(inputs, issue, target)
        native = getattr(model, "quantiles_for", None)
        if native is not None:
            return native(inputs, issue, target)
        return model.point_for(inputs, issue, target)
    except InsufficientDataError:
        return None


def _days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=k) for k in range((last - first).days + 1)]


def backtest_model(
    model: PointModel,
    inputs: Inputs,
    first_target: date,
    last_target: date,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Score rows (same columns as ``scores/``) for all targets in [first, last]."""
    prices = inputs.prices
    rows = []
    for h in HORIZONS:
        warm = first_target - timedelta(days=model.error_days + h)
        targets = _days(warm, last_target)
        points = Parallel(n_jobs=n_jobs)(
            delayed(_point)(model, inputs, t - timedelta(days=h), t) for t in targets
        )
        native = {t: p for t, p in zip(targets, points, strict=True) if isinstance(p, pd.DataFrame)}
        by_target = {
            t: (p["q50"] if isinstance(p, pd.DataFrame) else p)
            for t, p in zip(targets, points, strict=True)
            if p is not None
        }
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
                calibrate = getattr(model, "calibrate", None)
                if t in native and calibrate is not None:
                    calib = []
                    for k in range(model.error_days):
                        past = issue - timedelta(days=k)
                        if past in native:
                            actual = prices.reindex(native[past].index)
                            if actual.notna().mean() >= 0.9:
                                calib.append((native[past], actual))
                    if len(calib) < model.min_error_days:
                        calib = []
                    q = (
                        calibrate(native[t], calib)[["q10", "q50", "q90"]].to_numpy()
                        if calib
                        else np.full((len(slots), 3), np.nan)
                    )
                    status = "scored" if calib else "missed"
                elif t in native:
                    q = native[t][["q10", "q50", "q90"]].to_numpy()
                    status = "scored"
                else:
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
    weather: pd.DataFrame | None = None,
) -> BacktestResult:
    inputs = Inputs(normalize_index(prices), weather)
    parts = [backtest_model(m, inputs, first_target, last_target, n_jobs) for m in models]
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
    worst_path = out_dir / "worst_days.parquet"
    worst_days(res, WORST_DAYS_PER_MODEL).to_parquet(worst_path, index=False)
    return [summary_path, daily_path, worst_path]


WORST_DAYS_PER_MODEL = 12


def worst_days(res: BacktestResult, n: int) -> pd.DataFrame:
    """Quarter-hour rows of the ``n`` days with the largest MAE per model and horizon."""
    keys = (
        res.daily.sort_values("mae", ascending=False)
        .groupby(["model", "model_version", "horizon_days"])
        .head(n)[["model", "model_version", "horizon_days", "target_date"]]
    )
    cols = ["model", "model_version", "horizon_days", "target_date", "delivery_start_utc",
            "actual", "q10", "q50", "q90"]  # fmt: skip
    return res.scores.merge(keys, on=["model", "model_version", "horizon_days", "target_date"])[
        cols
    ]
