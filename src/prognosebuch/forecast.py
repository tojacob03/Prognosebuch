"""Issue the daily forecast for D+1 and D+2 and write it as immutable files.

Honesty guards (all enforced here, not only in the scheduler):
- The issue date is the current date in Europe/Berlin.
- Forecasts are only issued in the window [EARLIEST_ISSUE, DEADLINE) local time. The
  deadline is the auction gate closure (12:00); results are published around 12:45.
- If SMARD already shows any price for D+1, the run refuses (the forecast would be late).
- Models only receive an InfoSet, which hard-filters prices after the end of the issue day.
- Files are created with exclusive mode; an existing forecast is never overwritten. A
  second run on the same day only adds models that are still missing.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from prognosebuch import __version__
from prognosebuch.cheapest import METHOD as CHEAPEST_METHOD
from prognosebuch.cheapest import NEAR_EUR_MWH, WINDOW_HOURS, cheapest_window
from prognosebuch.history import load_actuals, price_history
from prognosebuch.models.bands import predict_with_errors
from prognosebuch.models.base import EARLIEST_ISSUE, InfoSet, weather_cutoff_utc
from prognosebuch.registry import HORIZONS, LIVE_MODELS, LiveModel
from prognosebuch.smard import ATTRIBUTION, PRICE_DE_LU, SmardClient
from prognosebuch.storage import (
    FORECAST_SCHEMA,
    FORECAST_SCHEMA_VERSION,
    dump_json,
    forecast_paths,
    sha256_hex,
    to_parquet_bytes,
    write_once,
)
from prognosebuch.timeutil import TZ, day_bounds_utc, local_iso
from prognosebuch.weather import ATTRIBUTION as WEATHER_ATTRIBUTION
from prognosebuch.weather import MODEL as WEATHER_MODEL
from prognosebuch.weather import POINTS as WEATHER_POINTS
from prognosebuch.weather import live_weather

DEADLINE = time(12, 0)
HISTORY_DAYS = 110  # fetched fresh from SMARD; older history comes from actuals/


class TooEarlyError(RuntimeError):
    pass


class DeadlinePassedError(RuntimeError):
    pass


class PricesAlreadyPublishedError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunInfo:
    code_commit: str | None
    workflow_run_url: str | None

    @classmethod
    def from_env(cls) -> RunInfo:
        server, repo, run_id = (
            os.environ.get("GITHUB_SERVER_URL"),
            os.environ.get("GITHUB_REPOSITORY"),
            os.environ.get("GITHUB_RUN_ID"),
        )
        url = f"{server}/{repo}/actions/runs/{run_id}" if server and repo and run_id else None
        return cls(code_commit=os.environ.get("GITHUB_SHA"), workflow_run_url=url)


@dataclass(frozen=True)
class ForecastResult:
    issue_date: date
    written: list[Path]
    skipped_existing: list[str]
    failed: dict[str, str]


def check_issue_window(now_utc: datetime) -> date:
    now_local = now_utc.astimezone(TZ)
    if now_local.time() < EARLIEST_ISSUE:
        raise TooEarlyError(f"local time {now_local:%H:%M} is before {EARLIEST_ISSUE:%H:%M}")
    if now_local.time() >= DEADLINE:
        raise DeadlinePassedError(
            f"local time {now_local:%H:%M} is at or after the deadline {DEADLINE:%H:%M}"
        )
    return now_local.date()


def build_forecast_frame(
    lm: LiveModel, info: InfoSet, issued_at_utc: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Forecast rows for D+1 and D+2, and the cheapest 3-hour window of each target day."""
    parts = []
    cheapest: dict[str, Any] = {}
    for h in HORIZONS:
        target = info.issue_date + timedelta(days=h)
        pred, errors = predict_with_errors(lm.model, info, target)
        cheapest[target.isoformat()] = cheapest_window(pred["q50"], errors)
        idx = pd.DatetimeIndex(pred.index)
        parts.append(
            pd.DataFrame(
                {
                    "schema_version": FORECAST_SCHEMA_VERSION,
                    "issue_date": info.issue_date,
                    "issued_at_utc": issued_at_utc,
                    "model": lm.model.name,
                    "model_version": lm.model.version,
                    "horizon_days": h,
                    "target_date": target,
                    "delivery_start_utc": idx,
                    "delivery_start_local": local_iso(idx),
                    "resolution": "PT15M",
                    "q10": pred["q10"].to_numpy(),
                    "q50": pred["q50"].to_numpy(),
                    "q90": pred["q90"].to_numpy(),
                }
            )
        )
    df = pd.concat(parts, ignore_index=True)
    if df[["q10", "q50", "q90"]].isna().any().any():
        raise ValueError(f"{lm.key}: forecast contains missing values")
    if not ((df["q10"] <= df["q50"]) & (df["q50"] <= df["q90"])).all():
        raise ValueError(f"{lm.key}: quantiles are not ordered")
    return df, cheapest


def run_forecast(
    root: Path,
    now_utc: datetime,
    client: SmardClient,
    run: RunInfo,
    models: tuple[LiveModel, ...] = LIVE_MODELS,
    weather_source: WeatherSource | None = None,
) -> ForecastResult:
    issue_date = now_utc.astimezone(TZ).date()
    todo = [
        lm
        for lm in models
        if lm.expected_on(issue_date)
        and not forecast_paths(root, issue_date, lm.model.name, lm.model.version)[0].exists()
    ]
    done = [lm.key for lm in models if lm not in todo and lm.expected_on(issue_date)]
    if not todo:
        return ForecastResult(issue_date, [], done, {})
    check_issue_window(now_utc)

    fetch_start = day_bounds_utc(issue_date - timedelta(days=HISTORY_DAYS))[0]
    fetch_end = day_bounds_utc(issue_date + timedelta(days=3))[1]
    fetched_at = pd.Timestamp(now_utc).floor("s")
    fresh = client.fetch(PRICE_DE_LU, fetch_start, fetch_end)

    next_day_start = day_bounds_utc(issue_date)[1]
    if (fresh.index >= next_day_start).any():
        raise PricesAlreadyPublishedError(
            f"SMARD already has prices for {issue_date + timedelta(days=1)}; refusing to issue"
        )
    archive = load_actuals(root)
    weather, weather_error = None, None
    if any(getattr(lm.model, "needs_weather", False) for lm in todo):
        try:
            weather = (weather_source or _live_weather)(root, issue_date)
        except Exception as exc:  # models that need weather fail; the others still run
            weather_error = f"{type(exc).__name__}: {exc}"
    info = InfoSet.cut(issue_date, price_history(archive, fresh), weather)
    issued_at = pd.Timestamp(now_utc).floor("s")

    written: list[Path] = []
    failed: dict[str, str] = {}
    for lm in todo:
        try:
            df, cheapest = build_forecast_frame(lm, info, issued_at)
        except Exception as exc:  # one failing model must not block the others
            failed[lm.key] = f"{type(exc).__name__}: {exc}"
            if weather_error and getattr(lm.model, "needs_weather", False):
                failed[lm.key] += f" (weather fetch failed: {weather_error})"
            continue
        pq_path, js_path = forecast_paths(root, issue_date, lm.model.name, lm.model.version)
        data = to_parquet_bytes(df, FORECAST_SCHEMA)
        manifest = build_manifest(lm, df, info, issued_at, fetched_at, fresh, archive, run, data)
        if getattr(lm.model, "needs_weather", False) and info.weather is not None:
            manifest["data_cutoffs"][WEATHER_KEY] = weather_cutoff_meta(info, fetched_at)
        manifest["cheapest_windows"] = {
            "method": CHEAPEST_METHOD,
            "window_hours": WINDOW_HOURS,
            "near_eur_mwh": NEAR_EUR_MWH,
            "by_target": cheapest,
        }
        write_once(pq_path, data)
        write_once(js_path, dump_json(manifest))
        written += [pq_path, js_path]
    return ForecastResult(issue_date, written, done, failed)


WeatherSource = Callable[[Path, date], pd.DataFrame]
WEATHER_KEY = f"open-meteo:{WEATHER_MODEL}:previous-runs"


def _live_weather(root: Path, issue_date: date) -> pd.DataFrame:
    return live_weather(root, issue_date)[0]


def weather_cutoff_meta(info: InfoSet, fetched_at: pd.Timestamp) -> dict[str, Any]:
    w = info.weather
    assert w is not None
    per_lead = {
        f"lead_{lead}_days_last_valid_utc": pd.Timestamp(g["valid_utc"].max()).isoformat()
        for lead, g in w.groupby("lead_days")
    }
    return {
        "description": "Weather forecasts (wind, solar, temperature), aggregated over "
        f"{len(WEATHER_POINTS)} points; D+1 uses lead 2 days, D+2 lead 3 days",
        "fetched_at_utc": fetched_at.isoformat(),
        "information_cutoff_utc": weather_cutoff_utc(info.issue_date).isoformat(),
        "last_available_at_utc": pd.Timestamp(w["available_at_utc"].max()).isoformat(),
        **per_lead,
        "n_values_used": len(w),
        "attribution": WEATHER_ATTRIBUTION,
    }


def build_manifest(
    lm: LiveModel,
    df: pd.DataFrame,
    info: InfoSet,
    issued_at: pd.Timestamp,
    fetched_at: pd.Timestamp,
    fresh: pd.Series,
    archive: pd.Series,
    run: RunInfo,
    parquet_bytes: bytes,
) -> dict[str, Any]:
    last_used = info.prices.index.max() if len(info.prices) else None
    first_used = info.prices.index.min() if len(info.prices) else None
    last_seen = fresh.index.max() if len(fresh) else None
    archive_last = archive.index.max() if len(archive) else None
    return {
        "schema_version": FORECAST_SCHEMA_VERSION,
        "model": lm.model.name,
        "model_version": lm.model.version,
        "model_description": lm.model.description,
        "issue_date": info.issue_date.isoformat(),
        "issued_at_utc": issued_at.isoformat(),
        "issued_at_local": issued_at.tz_convert(TZ).isoformat(),
        "target_dates": sorted({d.isoformat() for d in df["target_date"]}),
        "n_rows": len(df),
        "data_cutoffs": {
            PRICE_DE_LU.key: {
                "description": PRICE_DE_LU.description,
                "fetched_at_utc": fetched_at.isoformat(),
                "information_cutoff_utc": info.prices_known_until_utc.isoformat(),
                "last_value_used_utc": last_used.isoformat() if last_used is not None else None,
                "last_value_available_utc": last_seen.isoformat()
                if last_seen is not None
                else None,
                "first_value_used_utc": first_used.isoformat() if first_used is not None else None,
                "n_values_used": len(info.prices),
                "archive_last_value_utc": archive_last.isoformat()
                if archive_last is not None
                else None,
                "attribution": ATTRIBUTION,
            }
        },
        "package_version": __version__,
        "code_commit": run.code_commit,
        "workflow_run_url": run.workflow_run_url,
        "parquet_sha256": sha256_hex(parquet_bytes),
    }
