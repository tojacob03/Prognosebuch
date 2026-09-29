"""Downloadable exports (CSV for Power BI/Excel, Parquet, JSON) and their column dictionary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from prognosebuch.site.data import SiteData
from prognosebuch.smard import ATTRIBUTION
from prognosebuch.timeutil import TZ

REPO_URL = "https://github.com/tojacob03/Prognosebuch"
CSV_ACTUALS_SINCE = pd.Timestamp("2025-10-01", tz=TZ)


@dataclass(frozen=True)
class ExportFile:
    name: str
    rows: int
    bytes: int
    content: dict[str, str]  # lang -> description


# (table, column, type, unit, description en, description de)
DICTIONARY: list[tuple[str, str, str, str, str, str]] = [
    ("forecasts", "issue_date", "date", "", "Day D on which the forecast was issued (Europe/Berlin)", "Tag D, an dem die Prognose ausgegeben wurde (Europe/Berlin)"),
    ("forecasts", "issued_at_utc", "datetime", "UTC", "Exact issue time", "Genaue Ausgabezeit"),
    ("forecasts", "model_key", "text", "", "Model name and version, e.g. lear.v1", "Modellname und Version, z. B. lear.v1"),
    ("forecasts", "horizon_days", "integer", "days", "1 = forecast for the next day (D+1), 2 = for the day after (D+2)", "1 = Prognose für den nächsten Tag (D+1), 2 = für übermorgen (D+2)"),
    ("forecasts", "target_date", "date", "", "Delivery day (Europe/Berlin)", "Liefertag (Europe/Berlin)"),
    ("forecasts", "delivery_start_utc", "datetime", "UTC", "Start of the quarter-hour", "Beginn der Viertelstunde"),
    ("forecasts", "delivery_date_local", "date", "", "Delivery date in German local time", "Lieferdatum in deutscher Ortszeit"),
    ("forecasts", "delivery_time_local", "text", "HH:MM", "Start time in German local time (the repeated hour on the autumn switch appears twice)", "Startzeit in deutscher Ortszeit (die doppelte Stunde bei der Zeitumstellung im Herbst erscheint zweimal)"),
    ("forecasts", "q10", "decimal", "EUR/MWh", "10 % quantile: the price should be below this in about 1 of 10 cases", "10-%-Quantil: Der Preis sollte in etwa 1 von 10 Fällen darunter liegen"),
    ("forecasts", "q50", "decimal", "EUR/MWh", "Median, the point forecast", "Median, die Punktprognose"),
    ("forecasts", "q90", "decimal", "EUR/MWh", "90 % quantile", "90-%-Quantil"),
    ("actuals", "delivery_start_utc", "datetime", "UTC", "Start of the quarter-hour", "Beginn der Viertelstunde"),
    ("actuals", "delivery_date_local", "date", "", "Delivery date in German local time", "Lieferdatum in deutscher Ortszeit"),
    ("actuals", "delivery_time_local", "text", "HH:MM", "Start time in German local time", "Startzeit in deutscher Ortszeit"),
    ("actuals", "price_eur_mwh", "decimal", "EUR/MWh", "Day-ahead price DE-LU (Bundesnetzagentur | SMARD.de)", "Day-Ahead-Preis DE-LU (Bundesnetzagentur | SMARD.de)"),
    ("scores", "status", "text", "", "scored, missed (no forecast was issued) or incomplete", "scored, missed (keine Prognose ausgegeben) oder incomplete"),
    ("scores", "actual", "decimal", "EUR/MWh", "Published price", "Veröffentlichter Preis"),
    ("scores", "error", "decimal", "EUR/MWh", "actual minus q50", "Tatsächlich minus q50"),
    ("scores", "abs_error", "decimal", "EUR/MWh", "Absolute error; its mean is the MAE", "Absoluter Fehler; sein Mittel ist der MAE"),
    ("scores", "pinball", "decimal", "EUR/MWh", "Mean pinball loss over q10, q50, q90", "Mittlerer Pinball-Verlust über q10, q50, q90"),
    ("scores", "in_band", "true/false", "", "Actual price inside [q10, q90]", "Tatsächlicher Preis in [q10, q90]"),
    ("scores_daily", "mae", "decimal", "EUR/MWh", "Mean absolute error of the day", "Mittlerer absoluter Fehler des Tages"),
    ("scores_daily", "rmse", "decimal", "EUR/MWh", "Root mean squared error of the day", "Wurzel des mittleren quadratischen Fehlers"),
    ("scores_daily", "coverage_80", "decimal", "share", "Share of quarter-hours inside the 80 % band", "Anteil der Viertelstunden im 80-%-Band"),
    ("cheapest", "window_start_local", "text", "", "Start of the recommended 3-hour window", "Beginn des empfohlenen 3-Stunden-Fensters"),
    ("cheapest", "p_cheapest", "decimal", "share", "Stated probability that the window is the cheapest", "Angegebene Wahrscheinlichkeit, dass es das günstigste Fenster ist"),
    ("cheapest", "p_near", "decimal", "share", "Stated probability that it is within 5 EUR/MWh of the cheapest", "Angegebene Wahrscheinlichkeit, höchstens 5 €/MWh über dem günstigsten zu liegen"),
    ("cheapest", "hit", "true/false", "", "The window was in fact the cheapest", "Das Fenster war tatsächlich das günstigste"),
    ("cheapest", "regret_eur_mwh", "decimal", "EUR/MWh", "Actual mean price in the window minus the cheapest window", "Tatsächlicher Mittelpreis im Fenster minus günstigstes Fenster"),
]  # fmt: skip


def _local_cols(df: pd.DataFrame, col: str = "delivery_start_utc") -> pd.DataFrame:
    local = pd.DatetimeIndex(df[col]).tz_convert(TZ)
    out = df.copy()
    out["delivery_date_local"] = local.strftime("%Y-%m-%d")
    out["delivery_time_local"] = local.strftime("%H:%M")
    return out


def _utc_text(s: pd.Series) -> pd.Series:
    return (
        pd.DatetimeIndex(s)
        .tz_convert("UTC")
        .strftime("%Y-%m-%dT%H:%M:%SZ")
        .to_series(index=s.index)
    )


def _write(df: pd.DataFrame, out: Path, stem: str, parquet: bool = True) -> list[Path]:
    csv = df.copy()
    for c in csv.columns:
        if isinstance(csv[c].dtype, pd.DatetimeTZDtype):
            csv[c] = _utc_text(csv[c])
    p_csv = out / f"{stem}.csv"
    csv.to_csv(p_csv, index=False, float_format="%.4f")
    paths = [p_csv]
    if parquet:
        p_pq = out / f"{stem}.parquet"
        df.to_parquet(p_pq, index=False)
        paths.append(p_pq)
    return paths


def build_exports(sd: SiteData, out: Path) -> list[ExportFile]:
    out.mkdir(parents=True, exist_ok=True)
    files: list[ExportFile] = []

    def add(paths: list[Path], rows: int, de: str, en: str) -> None:
        for p in paths:
            files.append(ExportFile(p.name, rows, p.stat().st_size, {"de": de, "en": en}))

    if not sd.forecasts.empty:
        fc = _local_cols(sd.forecasts)
        cols = ["issue_date", "issued_at_utc", "model_key", "model", "model_version", "horizon_days",
                "target_date", "delivery_start_utc", "delivery_date_local", "delivery_time_local",
                "q10", "q50", "q90"]  # fmt: skip
        add(_write(fc[cols], out, "forecasts"), len(fc),
            "Alle live ausgegebenen Prognosen, je Viertelstunde",
            "All forecasts issued live, per quarter-hour")  # fmt: skip
    a = sd.actuals
    act = pd.DataFrame({"delivery_start_utc": a.index, "price_eur_mwh": a.to_numpy()})
    act = _local_cols(act)[
        ["delivery_start_utc", "delivery_date_local", "delivery_time_local", "price_eur_mwh"]
    ]
    recent = act[pd.DatetimeIndex(act["delivery_start_utc"]) >= CSV_ACTUALS_SINCE]
    add(_write(recent, out, "actuals", parquet=False), len(recent),
        "Tatsächliche Preise seit 1. Oktober 2025 (Viertelstunden-Auktion)",
        "Actual prices since 1 October 2025 (quarter-hour auction)")  # fmt: skip
    p = out / "actuals_since_2023.parquet"
    act.to_parquet(p, index=False)
    add([p], len(act), "Tatsächliche Preise seit 2023 (vor Oktober 2025 Stundenwerte)",
        "Actual prices since 2023 (hourly values before October 2025)")  # fmt: skip
    if not sd.scores.empty:
        sc = sd.scores.copy()
        sc["model_key"] = sc["model"] + ".v" + sc["model_version"]
        sc = _local_cols(sc)
        add(_write(sc, out, "scores"), len(sc),
            "Bewertung jeder Viertelstunde, inklusive verpasster Prognosen",
            "Score of every quarter-hour, including missed forecasts")  # fmt: skip
        daily = sd.daily_scores()
        add(
            _write(daily, out, "scores_daily", parquet=False),
            len(daily),
            "Fehler je Modell, Horizont und Liefertag",
            "Error per model, horizon and delivery day",
        )
    if not sd.cheapest_scores.empty:
        ch = sd.cheapest_scores.copy()
        ch["model_key"] = ch["model"] + ".v" + ch["model_version"]
        add(_write(ch, out, "cheapest", parquet=False), len(ch),
            "Empfohlene günstigste 3 Stunden und wie sie ausgingen",
            "Recommended cheapest 3 hours and how they turned out")  # fmt: skip
    dd = pd.DataFrame(
        DICTIONARY, columns=["table", "column", "type", "unit", "description_en", "description_de"]
    )
    p = out / "data_dictionary.csv"
    dd.to_csv(p, index=False)
    add([p], len(dd), "Spaltenbeschreibung aller Tabellen", "Column dictionary for all tables")
    for name, src in (("summary.json", sd.summary), ("backtest_summary.json", sd.backtest_summary)):
        if src is not None:
            p = out / name
            p.write_text(json.dumps(src, indent=1, ensure_ascii=False))
            de = (
                "Kennzahlen der Live-Bilanz"
                if name == "summary.json"
                else "Kennzahlen des Backtests (nicht Teil der Bilanz)"
            )
            en = (
                "Live track record metrics"
                if name == "summary.json"
                else "Backtest metrics (not part of the track record)"
            )
            add([p], 0, de, en)
    p = out / "latest.json"
    p.write_text(json.dumps(latest_json(sd), indent=1, ensure_ascii=False))
    add([p], 0, "Jüngste Prognose aller Modelle, günstigste Fenster, Bilanz",
        "Most recent forecast of all models, cheapest windows, scores")  # fmt: skip
    return files


def latest_json(sd: SiteData) -> dict[str, Any]:
    featured, rule = sd.featured_model()
    out: dict[str, Any] = {
        "generated_at_utc": pd.Timestamp(sd.now_utc).floor("s").isoformat(),
        "license": "CC BY 4.0",
        "attribution": f"Prognosebuch ({REPO_URL}); prices: {ATTRIBUTION}",
        "disclaimer": "Forecasts with uncertainty. Not investment advice, not a trading signal.",
        "unit": "EUR/MWh",
        "featured_model": featured,
        "featured_rule": rule,
        "issue_date": None,
        "forecasts": {},
    }
    issue = sd.latest_issue
    if issue is None:
        return out
    out["issue_date"] = issue.isoformat()
    for (d, key), m in sorted(sd.manifests.items()):
        if d != issue:
            continue
        rel = f"forecasts/{d:%Y}/{d:%m}/{d}/{key}"
        entry: dict[str, Any] = {
            "issued_at_utc": m["issued_at_utc"],
            "parquet_sha256": m["parquet_sha256"],
            "source_files": [
                f"{REPO_URL}/blob/main/{rel}.parquet",
                f"{REPO_URL}/blob/main/{rel}.json",
            ],
            "targets": {},
        }
        for tgt in m["target_dates"]:
            rows = sd.forecast_for(d, key, pd.Timestamp(tgt).date())
            entry["targets"][tgt] = {
                "cheapest_window": m.get("cheapest_windows", {}).get("by_target", {}).get(tgt),
                "quarter_hours": [
                    {
                        "start_utc": ts.isoformat(),
                        "start_local": ts.tz_convert(TZ).isoformat(),
                        "q10": round(float(lo), 2),
                        "q50": round(float(mid), 2),
                        "q90": round(float(hi), 2),
                    }
                    for ts, lo, mid, hi in zip(
                        pd.DatetimeIndex(rows["delivery_start_utc"]),
                        rows["q10"].to_numpy(),
                        rows["q50"].to_numpy(),
                        rows["q90"].to_numpy(),
                        strict=True,
                    )
                ],
            }
        out["forecasts"][key] = entry
    return out
