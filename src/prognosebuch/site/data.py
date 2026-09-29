"""Load the repository's public data for the website and the downloadable exports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from prognosebuch.evaluate import load_cheapest_scores, load_scores
from prognosebuch.history import load_actuals
from prognosebuch.registry import LIVE_MODELS, REFERENCE_MODEL_KEY
from prognosebuch.storage import read_parquet
from prognosebuch.timeutil import TZ, local_date

FEATURE_WINDOW_DAYS = 30
FEATURE_MIN_DAYS = 14


@dataclass
class SiteData:
    root: Path
    now_utc: datetime
    forecasts: pd.DataFrame
    manifests: dict[tuple[date, str], dict[str, Any]]
    actuals: pd.Series
    scores: pd.DataFrame
    cheapest_scores: pd.DataFrame
    summary: dict[str, Any] | None
    backtest_dir: Path | None
    backtest_summary: dict[str, Any] | None
    backtest_daily: pd.DataFrame | None
    backtest_worst: pd.DataFrame | None
    live_keys: list[str] = field(default_factory=list)
    first_issue: date | None = None

    # ------------------------------------------------------------------ derived

    @property
    def issue_dates(self) -> list[date]:
        return sorted({d for d, _ in self.manifests})

    @property
    def latest_issue(self) -> date | None:
        return self.issue_dates[-1] if self.manifests else None

    def daily_scores(self) -> pd.DataFrame:
        """Per model, horizon and target day: status, MAE, pinball, coverage."""
        s = self.scores
        if s.empty:
            return pd.DataFrame()
        s = s.assign(model_key=s["model"] + ".v" + s["model_version"])
        g = s.groupby(["model_key", "horizon_days", "target_date", "issue_date", "status"])
        d = g.agg(
            mae=("abs_error", "mean"),
            rmse=("sq_error", "mean"),
            pinball=("pinball", "mean"),
            coverage_80=("in_band", "mean"),
        ).reset_index()
        d["rmse"] = d["rmse"] ** 0.5
        d["target_date"] = pd.to_datetime(d["target_date"]).dt.date
        d["issue_date"] = pd.to_datetime(d["issue_date"]).dt.date
        return d

    def featured_model(self) -> tuple[str, str]:
        """Model shown by default, and why.

        Rule (stated on the site): the live model with the lowest D+1 MAE over the last 30
        scored days, once it has at least 14 of them; until then the reference rule.
        """
        d = self.daily_scores()
        if not d.empty:
            ok = d[(d["status"] == "scored") & (d["horizon_days"] == 1)]
            if not ok.empty:
                latest = max(ok["target_date"])
                w = ok[ok["target_date"] > latest - timedelta(days=FEATURE_WINDOW_DAYS)]
                counts = w.groupby("model_key")["target_date"].nunique()
                eligible = counts[counts >= FEATURE_MIN_DAYS].index
                if len(eligible):
                    mae = w[w["model_key"].isin(eligible)].groupby("model_key")["mae"].mean()
                    return str(mae.idxmin()), "best_recent"
        return REFERENCE_MODEL_KEY, "reference_until_enough_days"

    def forecast_for(self, issue: date, model_key: str, target: date) -> pd.DataFrame:
        f = self.forecasts
        if f.empty:
            return f
        sub = f[
            (f["issue_date"] == issue)
            & (f["model_key"] == model_key)
            & (f["target_date"] == target)
        ]
        return sub.sort_values("delivery_start_utc")

    def actual_day(self, d: date) -> pd.Series:
        from prognosebuch.timeutil import day_slots_utc

        return self.actuals.reindex(day_slots_utc(d))


def _read_all(files: list[Path]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    return pd.concat([read_parquet(f) for f in files], ignore_index=True)


def latest_backtest_dir(root: Path) -> Path | None:
    dirs = sorted(p for p in (root / "backtest").glob("*_*") if (p / "summary.json").exists())
    return dirs[-1] if dirs else None


def load_site_data(root: Path, now_utc: datetime) -> SiteData:
    fc_files = sorted((root / "forecasts").rglob("*.parquet"))
    forecasts = _read_all(fc_files)
    if not forecasts.empty:
        forecasts["model_key"] = forecasts["model"] + ".v" + forecasts["model_version"]
        forecasts["issue_date"] = pd.to_datetime(forecasts["issue_date"]).dt.date
        forecasts["target_date"] = pd.to_datetime(forecasts["target_date"]).dt.date
        forecasts["delivery_start_utc"] = pd.DatetimeIndex(forecasts["delivery_start_utc"]).as_unit(
            "ns"
        )
    manifests: dict[tuple[date, str], dict[str, Any]] = {}
    for js in sorted((root / "forecasts").rglob("*.json")):
        m = json.loads(js.read_text())
        manifests[(date.fromisoformat(m["issue_date"]), f"{m['model']}.v{m['model_version']}")] = m
    summary_path = root / "scores" / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    scores = load_scores(root)
    cheapest = load_cheapest_scores(root)
    bdir = latest_backtest_dir(root)
    b_summary = b_daily = b_worst = None
    if bdir is not None:
        b_summary = json.loads((bdir / "summary.json").read_text())
        if (bdir / "daily.csv").exists():
            b_daily = pd.read_csv(bdir / "daily.csv", parse_dates=["target_date"])
            b_daily["target_date"] = b_daily["target_date"].dt.date
        if (bdir / "worst_days.parquet").exists():
            b_worst = read_parquet(bdir / "worst_days.parquet")
    return SiteData(
        root=root,
        now_utc=now_utc,
        forecasts=forecasts,
        manifests=manifests,
        actuals=load_actuals(root),
        scores=scores,
        cheapest_scores=cheapest,
        summary=summary,
        backtest_dir=bdir,
        backtest_summary=b_summary,
        backtest_daily=b_daily,
        backtest_worst=b_worst,
        live_keys=[lm.key for lm in LIVE_MODELS],
        first_issue=min([lm.live_since for lm in LIVE_MODELS] + [d for d, _ in manifests]),
    )


def local_today(now_utc: datetime) -> date:
    return local_date(pd.Timestamp(now_utc))


def local_hhmm(ts: pd.Timestamp | str) -> str:
    return pd.Timestamp(ts).tz_convert(TZ).strftime("%H:%M")
