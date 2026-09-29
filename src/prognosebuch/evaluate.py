"""Score issued forecasts against published prices and keep the book complete.

For every target day whose prices are fully published, and every live model and horizon
that was due, one row per quarter-hour is written to ``scores/``. A forecast that does not
exist is recorded with status ``missed``; missed days are never dropped from the book.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prognosebuch.metrics import diebold_mariano, mae, mean_pinball, rmse, skill
from prognosebuch.registry import HORIZONS, LIVE_MODELS, REFERENCE_MODEL_KEY, LiveModel
from prognosebuch.smard import PRICE_DE_LU, SmardClient
from prognosebuch.storage import (
    ACTUALS_SCHEMA,
    SCORE_SCHEMA,
    SCORE_SCHEMA_VERSION,
    actuals_path,
    dump_json,
    forecast_paths,
    read_parquet,
    score_path,
    to_parquet_bytes,
    write_once,
)
from prognosebuch.timeutil import TZ, day_bounds_utc, day_slots_utc, local_date, local_iso

ACTUALS_LOOKBACK_DAYS = 21
WINDOWS: dict[str, int | None] = {"7d": 7, "30d": 30, "90d": 90, "all": None}
MIN_DM_DAYS = 10


# --------------------------------------------------------------------------- actuals


def update_actuals(root: Path, client: SmardClient, now_utc: datetime) -> list[Path]:
    """Merge the latest SMARD prices into the monthly actuals files (newer values win)."""
    today = local_date(pd.Timestamp(now_utc))
    start = day_bounds_utc(today - timedelta(days=ACTUALS_LOOKBACK_DAYS))[0]
    end = day_bounds_utc(today + timedelta(days=2))[1]
    fresh = client.fetch(PRICE_DE_LU, start, end)
    if fresh.empty:
        return []
    months = pd.Series(
        pd.DatetimeIndex(fresh.index).tz_convert(TZ).strftime("%Y-%m"), index=fresh.index
    )
    changed: list[Path] = []
    for month, idx in fresh.groupby(months).groups.items():
        path = actuals_path(root, str(month))
        new = fresh.loc[idx]
        old = load_actuals_file(path)
        merged = pd.concat([old, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        if old.equals(merged):
            continue
        df = pd.DataFrame(
            {
                "delivery_start_utc": merged.index,
                "delivery_start_local": local_iso(pd.DatetimeIndex(merged.index)),
                "price_eur_mwh": merged.to_numpy(),
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(to_parquet_bytes(df, ACTUALS_SCHEMA))
        changed.append(path)
    return changed


def load_actuals_file(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(
            [], index=pd.DatetimeIndex([], tz="UTC", name="delivery_start_utc"), dtype="float64"
        )
    df = read_parquet(path)
    s = pd.Series(
        df["price_eur_mwh"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(df["delivery_start_utc"], name="delivery_start_utc"),
    )
    s.index = pd.DatetimeIndex(s.index).as_unit("ns")
    return s


def load_actuals(root: Path) -> pd.Series:
    files = sorted((root / "actuals" / "day_ahead_price_de_lu").glob("*.parquet"))
    parts = [load_actuals_file(f) for f in files]
    if not parts:
        return load_actuals_file(Path("/nonexistent"))
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="last")]


def day_is_complete(actuals: pd.Series, d: date) -> bool:
    slots = day_slots_utc(d)
    return bool(actuals.reindex(slots).notna().all())


# --------------------------------------------------------------------------- scoring


def load_forecast(root: Path, lm: LiveModel, issue_date: date) -> pd.DataFrame | None:
    path = forecast_paths(root, issue_date, lm.model.name, lm.model.version)[0]
    if not path.exists():
        return None
    return read_parquet(path)


def score_target_day(
    root: Path,
    target: date,
    actuals: pd.Series,
    scored_at: pd.Timestamp,
    models: tuple[LiveModel, ...] = LIVE_MODELS,
) -> pd.DataFrame | None:
    slots = day_slots_utc(target)
    y = actuals.reindex(slots).to_numpy(dtype=float)
    frames = []
    for lm in models:
        for h in HORIZONS:
            issue = target - timedelta(days=h)
            if not lm.expected_on(issue):
                continue
            fc = load_forecast(root, lm, issue)
            q = np.full((len(slots), 3), np.nan)
            status = "missed"
            if fc is not None:
                sub = fc[(fc["target_date"] == target) & (fc["horizon_days"] == h)]
                sub = sub.set_index(pd.DatetimeIndex(sub["delivery_start_utc"]).as_unit("ns"))
                q = sub.reindex(slots)[["q10", "q50", "q90"]].to_numpy(dtype=float)
                status = "scored" if not np.isnan(q).any() else "incomplete"
            err = y - q[:, 1]
            frames.append(
                pd.DataFrame(
                    {
                        "schema_version": SCORE_SCHEMA_VERSION,
                        "target_date": target,
                        "delivery_start_utc": slots,
                        "delivery_start_local": local_iso(slots),
                        "model": lm.model.name,
                        "model_version": lm.model.version,
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
                        "in_band": pd.array(
                            [
                                None if np.isnan(lo) else bool(lo <= yy <= hi)
                                for yy, lo, hi in zip(y, q[:, 0], q[:, 2], strict=True)
                            ],
                            dtype="boolean",
                        ),
                        "scored_at_utc": scored_at,
                    }
                )
            )
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class EvaluateResult:
    actuals_changed: list[Path]
    scored_days: list[date]
    pending_days: list[date]
    summary_path: Path | None


def run_evaluate(
    root: Path,
    now_utc: datetime,
    client: SmardClient | None,
    models: tuple[LiveModel, ...] = LIVE_MODELS,
) -> EvaluateResult:
    changed = update_actuals(root, client, now_utc) if client is not None else []
    actuals = load_actuals(root)
    first_target = min(lm.live_since for lm in models) + timedelta(days=min(HORIZONS))
    last_target = local_date(pd.Timestamp(now_utc)) + timedelta(days=1)
    scored_at = pd.Timestamp(now_utc).floor("s")
    scored, pending = [], []
    d = first_target
    while d <= last_target:
        path = score_path(root, d)
        if not path.exists():
            if day_is_complete(actuals, d):
                df = score_target_day(root, d, actuals, scored_at, models)
                if df is not None:
                    write_once(path, to_parquet_bytes(df, SCORE_SCHEMA))
                    scored.append(d)
            else:
                pending.append(d)
        d += timedelta(days=1)
    scores = load_scores(root)
    summary_path = None
    if not scores.empty:
        summary_path = root / "scores" / "summary.json"
        summary_path.write_bytes(dump_json(summarize(scores, now_utc)))
    return EvaluateResult(changed, scored, pending, summary_path)


def load_scores(root: Path) -> pd.DataFrame:
    files = sorted((root / "scores").glob("*/*/*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([read_parquet(f) for f in files], ignore_index=True)


# --------------------------------------------------------------------------- summary


def _metrics(g: pd.DataFrame) -> dict[str, float | int | None]:
    ok = g[g["status"] == "scored"]
    if ok.empty:
        return {"n_slots": 0, "mae": None, "rmse": None, "pinball": None, "coverage_80": None}
    err = ok["error"].to_numpy(dtype=float)
    return {
        "n_slots": len(ok),
        "mae": round(mae(err), 3),
        "rmse": round(rmse(err), 3),
        "pinball": round(float(ok["pinball"].mean()), 3),
        "coverage_80": round(float(ok["in_band"].astype(float).mean()), 3),
    }


def _segments(df: pd.DataFrame, spike_threshold: float) -> dict[str, pd.DataFrame]:
    weekend = pd.to_datetime(df["target_date"]).dt.weekday >= 5
    return {
        "all": df,
        "weekday": df[~weekend],
        "weekend": df[weekend],
        "negative_price": df[df["actual"] < 0],
        "price_spike": df[df["actual"] >= spike_threshold],
    }


def summarize(scores: pd.DataFrame, now_utc: datetime) -> dict[str, Any]:
    s = scores.copy()
    s["model_key"] = s["model"] + ".v" + s["model_version"]
    s["target_date"] = pd.to_datetime(s["target_date"]).dt.date
    latest = max(s["target_date"])
    out: dict[str, Any] = {
        "generated_at_utc": pd.Timestamp(now_utc).floor("s").isoformat(),
        "latest_target_date": latest.isoformat(),
        "reference_model": REFERENCE_MODEL_KEY,
        "definitions": {
            "error": "actual - q50 (EUR/MWh)",
            "pinball": "mean pinball loss over q10, q50, q90",
            "coverage_80": "share of actual prices inside [q10, q90]",
            "skill_mae": "1 - MAE(model) / MAE(reference) on the same scored quarter-hours",
            "price_spike": "quarter-hours whose actual price is at or above the 90th "
            "percentile of all actual prices in the window",
            "dm_test": "Diebold-Mariano (HLN-corrected, two-sided) on daily MAE, "
            "negative statistic = first model better",
        },
        "windows": {},
        "missed": [],
    }
    for wname, days in WINDOWS.items():
        w = s if days is None else s[s["target_date"] > latest - timedelta(days=days)]
        prices = w.drop_duplicates("delivery_start_utc")["actual"]
        spike = float(np.nanquantile(prices.to_numpy(dtype=float), 0.9))
        rows = []
        for (key, h), g in w.groupby(["model_key", "horizon_days"]):
            hz = int(str(h))
            ref = w[(w["model_key"] == REFERENCE_MODEL_KEY) & (w["horizon_days"] == hz)]
            row: dict[str, Any] = {
                "model": str(key),
                "horizon_days": hz,
                "days_due": int(g["target_date"].nunique()),
                "days_forecast": int(g.loc[g["status"] == "scored", "target_date"].nunique()),
            }
            for seg, sg in _segments(g, spike).items():
                m = _metrics(sg)
                paired = sg.merge(
                    ref[["delivery_start_utc", "status", "abs_error"]],
                    on="delivery_start_utc",
                    suffixes=("", "_ref"),
                )
                paired = paired[(paired["status"] == "scored") & (paired["status_ref"] == "scored")]
                m["skill_mae"] = (
                    round(skill(paired["abs_error"].mean(), paired["abs_error_ref"].mean()), 4)
                    if len(paired)
                    else None
                )
                row[seg] = m
            rows.append(row)
        out["windows"][wname] = {
            "from": (min(w["target_date"])).isoformat(),
            "to": latest.isoformat(),
            "spike_threshold_eur_mwh": round(spike, 2),
            "models": rows,
            "dm_tests": _dm_tests(w),
        }
    missed = s[s["status"] != "scored"].drop_duplicates(
        ["model_key", "horizon_days", "target_date"]
    )
    out["missed"] = [
        {
            "model": str(r["model_key"]),
            "horizon_days": int(r["horizon_days"]),
            "target_date": str(r["target_date"]),
            "issue_date": str(pd.Timestamp(r["issue_date"]).date()),
            "status": str(r["status"]),
        }
        for r in missed.to_dict("records")
    ]
    return out


def _dm_tests(w: pd.DataFrame) -> list[dict[str, Any]]:
    ok = w[w["status"] == "scored"]
    daily = ok.groupby(["model_key", "horizon_days", "target_date"])["abs_error"].mean()
    res = []
    keys = sorted(ok["model_key"].unique())
    for h in sorted(ok["horizon_days"].unique()):
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                try:
                    la, lb = daily.loc[(a, h)], daily.loc[(b, h)]
                except KeyError:
                    continue
                both = pd.concat([la, lb], axis=1, join="inner").dropna()
                entry: dict[str, Any] = {
                    "model_a": a,
                    "model_b": b,
                    "horizon_days": int(h),
                    "n_days": len(both),
                }
                if len(both) >= MIN_DM_DAYS:
                    r = diebold_mariano(both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy(), h)
                    entry |= {
                        "statistic": round(r.statistic, 3),
                        "p_value": round(r.p_value, 4),
                        "mean_daily_mae_diff": round(r.mean_loss_diff, 3),
                    }
                else:
                    entry["note"] = f"fewer than {MIN_DM_DAYS} paired days"
                res.append(entry)
    return res
