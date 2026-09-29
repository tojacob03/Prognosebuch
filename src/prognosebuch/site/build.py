"""Build the static website (German at /, English at /en/) into an output directory."""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt

from prognosebuch.cheapest import WINDOW_SLOTS
from prognosebuch.registry import HORIZONS, REFERENCE_MODEL_KEY
from prognosebuch.site import charts
from prognosebuch.site.data import SiteData, load_site_data, local_hhmm, local_today
from prognosebuch.site.explain import explain_day
from prognosebuch.site.exports import REPO_URL, ExportFile, build_exports
from prognosebuch.site.i18n import (
    LANGS,
    PAGES,
    T,
    long_date,
    num,
    pct,
    short_date,
    t,
)
from prognosebuch.timeutil import TZ

HERE = Path(__file__).parent
WINDOW_KEYS = ("7d", "30d", "90d", "all")
BOOK_ROWS = 14
WORST_LIVE = 8
WORST_BACKTEST = 4


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(HERE / "templates"),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env


def _rel(from_page: str, lang: str, to_page: str, to_lang: str | None = None) -> str:
    depth = PAGES[from_page][lang].count("/")
    return "../" * depth + PAGES[to_page][to_lang or lang]


def _prefix(page: str, lang: str) -> str:
    return "../" * PAGES[page][lang].count("/")


# ---------------------------------------------------------------------------- pieces


def _tooltip_rows(df: pd.DataFrame, actual: pd.Series | None) -> list[list[Any]]:
    local = pd.DatetimeIndex(df["delivery_start_utc"]).tz_convert(TZ)
    act = None if actual is None else actual.to_numpy(dtype=float)
    rows = []
    for i, ts in enumerate(local):
        a = None if act is None or np.isnan(act[i]) else round(float(act[i]), 1)
        rows.append(
            [
                ts.strftime("%H:%M"),
                round(float(df["q10"].iloc[i]), 1),
                round(float(df["q50"].iloc[i]), 1),
                round(float(df["q90"].iloc[i]), 1),
                a,
            ]
        )
    return rows


def _highlight(df: pd.DataFrame, window: dict[str, Any] | None) -> tuple[int, int] | None:
    if not window:
        return None
    idx = pd.DatetimeIndex(df["delivery_start_utc"])
    hits = np.nonzero(idx == pd.Timestamp(window["start_utc"]))[0]
    return (int(hits[0]), WINDOW_SLOTS) if len(hits) else None


def day_panel(
    sd: SiteData,
    issue: date,
    model_key: str,
    h: int,
    lang: str,
    chart_id: str,
    frames: tuple[charts.Frame, ...] = (charts.WIDE, charts.NARROW),
) -> dict[str, Any] | None:
    target = issue + timedelta(days=h)
    df = sd.forecast_for(issue, model_key, target)
    if df.empty:
        return None
    manifest = sd.manifests.get((issue, model_key), {})
    window = manifest.get("cheapest_windows", {}).get("by_target", {}).get(target.isoformat())
    day_actual = sd.actual_day(target)
    actual = day_actual if day_actual.notna().any() else None
    hl = _highlight(df, window)
    title = f"{t('median', lang)} / {t('band', lang)}, {long_date(target, lang)}"
    svgs = [
        charts.day_chart(
            df,
            lang,
            frame=fr,
            actual=actual,
            highlight=hl,
            highlight_label=t("cheapest_3h", lang),
            title=title,
            chart_id=f"{chart_id}-{k}",
        )
        for k, fr in enumerate(frames)
    ]
    q50 = df["q50"].to_numpy(dtype=float)
    local = pd.DatetimeIndex(df["delivery_start_utc"]).tz_convert(TZ)
    panel: dict[str, Any] = {
        "h": h,
        "target": target,
        "target_long": long_date(target, lang),
        "target_short": short_date(target, lang),
        "svgs": svgs,
        "chart_id": chart_id,
        "tooltip": json.dumps(_tooltip_rows(df, actual)),
        "day_mean": num(float(q50.mean()), 0, lang),
        "day_mean_ct": num(float(q50.mean()) / 10, 1, lang),
        "min": num(float(q50.min()), 0, lang),
        "min_at": local[int(np.argmin(q50))].strftime("%H:%M"),
        "max": num(float(q50.max()), 0, lang),
        "max_at": local[int(np.argmax(q50))].strftime("%H:%M"),
        "window": None,
        "mae": None,
        "table": [
            (ts.strftime("%H:%M"), num(lo, 1, lang), num(mid, 1, lang), num(hi, 1, lang))
            for ts, lo, mid, hi in zip(
                local, df["q10"].to_numpy(), q50, df["q90"].to_numpy(), strict=True
            )
        ],
    }
    if window:
        panel["window"] = {
            "start": local_hhmm(window["start_utc"]),
            "end": local_hhmm(window["end_utc"]),
            "p_cheapest": pct(window["p_cheapest"], lang),
            "p_near": pct(window["p_near"], lang),
            "pc100": num(100 * window["p_cheapest"], 0, lang),
            "pn100": num(100 * window["p_near"], 0, lang),
            "expected_ct": num(window["expected_mean_eur_mwh"] / 10, 1, lang),
            "day_ct": num(window["day_mean_eur_mwh"] / 10, 1, lang),
            "alternatives": [
                {
                    "start": local_hhmm(a["start_utc"]),
                    "end": local_hhmm(a["end_utc"]),
                    "p": pct(a["p_cheapest"], lang),
                }
                for a in window.get("most_likely", [])
            ],
        }
    if actual is not None and actual.notna().all():
        panel["mae"] = num(float(np.mean(np.abs(actual.to_numpy() - q50))), 1, lang)
    return panel


def _pick_model(sd: SiteData, issue: date, preferred: str) -> str | None:
    have = [k for (d, k) in sd.manifests if d == issue]
    for k in (preferred, REFERENCE_MODEL_KEY, *sorted(have)):
        if k in have:
            return k
    return None


def book_rows(sd: SiteData, lang: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Ledger of the most recent delivery days: D+1 status/MAE per live model."""
    keys = sd.live_keys
    daily = sd.daily_scores()
    lookup: dict[tuple[str, date], tuple[str, float]] = {}
    if not daily.empty:
        d1 = daily[daily["horizon_days"] == 1]
        for r in d1.to_dict("records"):
            lookup[(str(r["model_key"]), r["target_date"])] = (str(r["status"]), float(r["mae"]))
    if sd.first_issue is None:
        return keys, []
    first_target = sd.first_issue + timedelta(days=1)
    last_issue = sd.latest_issue or sd.first_issue
    last_target = max(last_issue + timedelta(days=1), first_target)
    today = local_today(sd.now_utc)
    rows: list[dict[str, Any]] = []
    d = last_target
    while d >= first_target and len(rows) < BOOK_ROWS:
        issue = d - timedelta(days=1)
        cells = []
        for k in keys:
            status, mae = lookup.get((k, d), ("", float("nan")))
            if status == "scored":
                cells.append({"cls": "", "text": num(mae, 1, lang)})
            elif status in ("missed", "incomplete") or (
                (issue, k) not in sd.manifests and issue < today
            ):
                cells.append({"cls": "red", "text": t("status_missed", lang)})
            else:
                cells.append({"cls": "muted", "text": t("status_pending", lang)})
        issued = sd.manifests.get((issue, REFERENCE_MODEL_KEY)) or next(
            (m for (di, _), m in sd.manifests.items() if di == issue), None
        )
        rows.append(
            {
                "day": short_date(d, lang),
                "entry": (d - first_target).days + 1,
                "issued": local_hhmm(issued["issued_at_utc"]) if issued else "–",
                "issue_link": f"{REPO_URL}/tree/main/forecasts/{issue:%Y}/{issue:%m}/{issue}"
                if issued
                else None,
                "cells": cells,
            }
        )
        d -= timedelta(days=1)
    return keys, rows


def metric_tables(summary: dict[str, Any], lang: str) -> dict[str, Any]:
    """Per window and horizon: rows of headline metrics; segment tables; DM tests."""
    out: dict[str, Any] = {}
    for w in WINDOW_KEYS:
        win = summary["windows"].get(w)
        if not win:
            continue
        per_h: dict[int, list[dict[str, Any]]] = {}
        seg_h: dict[int, list[dict[str, Any]]] = {}
        for r in win["models"]:
            a = r["all"]
            row = {
                "model": r["model"],
                "is_ref": r["model"] == REFERENCE_MODEL_KEY,
                "days": f"{r['days_due']} / {r['days_forecast']}",
                "missed": r["days_due"] - r["days_forecast"],
                "mae": num(a["mae"], 1, lang),
                "mae_raw": a["mae"] if a["mae"] is not None else 1e9,
                "rmse": num(a["rmse"], 1, lang),
                "skill": num(a["skill_mae"], 2, lang, sign=True) if a["skill_mae"] else "0",
                "skill_neg": (a["skill_mae"] or 0) < 0,
                "pinball": num(a["pinball"], 1, lang),
                "cov": pct(a["coverage_80"], lang),
            }
            per_h.setdefault(r["horizon_days"], []).append(row)
            seg_h.setdefault(r["horizon_days"], []).append(
                {
                    "model": r["model"],
                    "is_ref": row["is_ref"],
                    "cells": [
                        (num(r[s]["mae"], 1, lang), r[s]["n_slots"])
                        for s in ("all", "weekday", "weekend", "negative_price", "price_spike")
                    ],
                }
            )
        for rows in per_h.values():
            rows.sort(key=lambda x: (not x["is_ref"], x["mae_raw"]))
        dm = []
        for d in win.get("dm_tests", []):
            if "p_value" not in d:
                verdict = t("dm_few", lang, n=d["n_days"])
            elif d["p_value"] < 0.05:
                better, worse = (
                    (d["model_a"], d["model_b"])
                    if d["statistic"] < 0
                    else (d["model_b"], d["model_a"])
                )
                verdict = t("dm_better", lang, a=better, b=worse)
            else:
                verdict = t("dm_none", lang)
            p = d.get("p_value")
            dm.append(
                {
                    "a": d["model_a"],
                    "b": d["model_b"],
                    "h": d["horizon_days"],
                    "n": d["n_days"],
                    "p": ("< 0,0001" if lang == "de" else "< 0.0001")
                    if p is not None and p < 1e-4
                    else num(p, 4, lang)
                    if p is not None
                    else "–",
                    "verdict": verdict,
                }
            )
        out[w] = {
            "label": t(f"w_{w}", lang),
            "from": win["from"],
            "to": win["to"],
            "spike": num(win["spike_threshold_eur_mwh"], 0, lang),
            "horizons": {h: per_h.get(h, []) for h in HORIZONS},
            "segments": {h: seg_h.get(h, []) for h in HORIZONS},
            "dm": dm,
        }
    return out


def cheapest_table(summary: dict[str, Any] | None, lang: str) -> list[dict[str, Any]]:
    if not summary or "cheapest_windows" not in summary:
        return []
    rows = []
    for r in summary["cheapest_windows"]["windows"].get("all", []):
        if not r.get("days_scored"):
            continue
        rows.append(
            {
                "model": r["model"],
                "h": r["horizon_days"],
                "days": r["days_scored"],
                "hit": pct(r["hit_rate"], lang),
                "p_stated": pct(r["mean_p_cheapest"], lang),
                "near": pct(r["near_rate"], lang),
                "pn_stated": pct(r["mean_p_near"], lang),
                "regret": num(r["mean_regret_eur_mwh"] / 10, 2, lang),
                "saving": num(r["saving_vs_day_mean_eur_mwh"] / 10, 2, lang),
            }
        )
    return rows


def failure_cards(
    sd: SiteData, lang: str, source: str, frame: charts.Frame = charts.SMALL
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    featured, _ = sd.featured_model()
    if source == "live":
        daily = sd.daily_scores()
        if daily.empty:
            return []
        ok = daily[(daily["status"] == "scored") & (daily["horizon_days"] == 1)]
        ok = ok[ok["model_key"] == featured]
        top = ok.sort_values("mae", ascending=False).head(WORST_LIVE)
        for r in top.to_dict("records"):
            issue, key, target = r["issue_date"], str(r["model_key"]), r["target_date"]
            df = sd.forecast_for(issue, key, target)
            cards.append(_card(sd, df, key, 1, target, float(r["mae"]), lang, frame, "live"))
        return cards
    w = sd.backtest_worst
    if w is None or w.empty:
        return []
    w = w.assign(model_key=w["model"] + ".v" + w["model_version"].astype(str))
    w["target_date"] = pd.to_datetime(w["target_date"]).dt.date
    for key in dict.fromkeys(["lear.v1", REFERENCE_MODEL_KEY]):
        sub = w[(w["model_key"] == key) & (w["horizon_days"] == 1)]
        if sub.empty:
            continue
        mae = (sub["actual"] - sub["q50"]).abs().groupby(sub["target_date"]).mean()
        worst = mae.sort_values(ascending=False).head(WORST_BACKTEST)
        for target, m in zip(list(worst.index), worst.to_numpy(), strict=True):
            df = sub[sub["target_date"] == target].sort_values("delivery_start_utc")
            cards.append(_card(sd, df, key, 1, target, float(m), lang, frame, "bt"))
    return cards


def _card(
    sd: SiteData,
    df: pd.DataFrame,
    key: str,
    h: int,
    target: date,
    mae: float,
    lang: str,
    frame: charts.Frame,
    prefix: str,
) -> dict[str, Any]:
    df = df.copy()
    df["delivery_start_utc"] = pd.DatetimeIndex(df["delivery_start_utc"]).as_unit("ns")
    actual = sd.actual_day(target)
    q50 = pd.Series(
        df["q50"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["delivery_start_utc"])
    )
    cid = f"{prefix}-{key}-{target}".replace(".", "-")
    return {
        "date": long_date(target, lang),
        "head": t(
            "fail_card_head",
            lang,
            model=key,
            horizon=t(f"h_short_{h}", lang),
            mae=num(mae, 0, lang),
        ),
        "svg": charts.day_chart(
            df, lang, frame=frame, actual=actual, title=long_date(target, lang), chart_id=cid
        ),
        "hints": explain_day(sd.actuals, target, h, q50, lang),
    }


def backtest_context(sd: SiteData, lang: str) -> dict[str, Any] | None:
    s = sd.backtest_summary
    if not s:
        return None
    period = s.get("period", {})
    first = date.fromisoformat(period.get("first_target", s["windows"]["all"]["from"]))
    last = date.fromisoformat(period.get("last_target", s["windows"]["all"]["to"]))
    tables = metric_tables(s, lang)
    return {
        "lede": t(
            "backtest_lede",
            lang,
            n=(last - first).days + 1,
            from_=long_date(first, lang, weekday=False),
            to=long_date(last, lang, weekday=False),
        ),
        "all": tables.get("all"),
        "link": f"{REPO_URL}/tree/main/backtest",
    }


def mae_chart(sd: SiteData, lang: str) -> str:
    daily = sd.daily_scores()
    if daily.empty:
        return ""
    ok = daily[(daily["status"] == "scored") & (daily["horizon_days"] == 1)]
    series = {
        str(k): g.set_index("target_date")["mae"].sort_index() for k, g in ok.groupby("model_key")
    }
    featured, _ = sd.featured_model()
    return charts.line_chart(
        series,
        lang,
        emphasis=featured,
        reference=REFERENCE_MODEL_KEY,
        title=t("chart_mae_title", lang),
        chart_id="mae",
    )


# ---------------------------------------------------------------------------- pages


def page_context(sd: SiteData, page: str, lang: str, files: list[ExportFile]) -> dict[str, Any]:
    featured, rule = sd.featured_model()
    ctx: dict[str, Any] = {
        "lang": lang,
        "page": page,
        "T": {k: v[lang] for k, v in T.items()},
        "prefix": _prefix(page, lang),
        "nav": [(p, _rel(page, lang, p), t(f"nav_{p}", lang)) for p in PAGES],
        "other_lang": "en" if lang == "de" else "de",
        "other_url": _rel(page, lang, page, "en" if lang == "de" else "de"),
        "repo": REPO_URL,
        "built": pd.Timestamp(sd.now_utc).tz_convert(TZ).strftime("%Y-%m-%d %H:%M"),
        "featured": featured,
        "featured_rule": t("model_rule_best" if rule == "best_recent" else "model_rule_ref", lang),
        "url": lambda p: _rel(page, lang, p),
    }
    issue = sd.latest_issue
    first_issue = sd.first_issue or local_today(sd.now_utc)
    if page in ("home", "cheapest"):
        panels = []
        if issue is not None:
            key = _pick_model(sd, issue, featured)
            if key is not None:
                ctx["shown_model"] = key
                m = sd.manifests[(issue, key)]
                issued_local = pd.Timestamp(m["issued_at_utc"]).tz_convert(TZ)
                ctx["issued"] = f"{long_date(issue, lang)}, {issued_local:%H:%M}"
                for h in HORIZONS:
                    frames = (charts.WIDE, charts.NARROW) if page == "home" else (charts.NARROW,)
                    p = day_panel(sd, issue, key, h, lang, f"{page}-d{h}", frames)
                    if p:
                        panels.append(p)
        ctx["panels"] = panels
        ctx["first_issue"] = long_date(first_issue, lang)
    if page == "home":
        ctx["book_keys"], ctx["book"] = book_rows(sd, lang)
        ctx["issue_link"] = (
            f"{REPO_URL}/tree/main/forecasts/{issue:%Y}/{issue:%m}/{issue}" if issue else None
        )
    if page == "record":
        ctx["tables"] = metric_tables(sd.summary, lang) if sd.summary else {}
        ctx["first_target"] = long_date(first_issue + timedelta(days=1), lang)
        ctx["mae_chart"] = mae_chart(sd, lang)
        ctx["cheap_rows"] = cheapest_table(sd.summary, lang)
        ctx["missed"] = [
            {**m, "day": short_date(date.fromisoformat(m["target_date"]), lang)}
            for m in (sd.summary or {}).get("missed", [])
        ]
        ctx["backtest"] = backtest_context(sd, lang)
    if page == "failures":
        ctx["live_cards"] = failure_cards(sd, lang, "live")
        ctx["bt_cards"] = failure_cards(sd, lang, "bt")
    if page == "cheapest":
        ctx["cheap_rows"] = cheapest_table(sd.summary, lang)
    if page == "method":
        md = MarkdownIt("commonmark", {"typographer": True}).enable("table")
        src = HERE / "content" / "methodik.de.md" if lang == "de" else sd.root / "METHODOLOGY.md"
        text = src.read_text()
        text = text.split("\n", 1)[1] if text.startswith("# ") else text
        ctx["method_html"] = md.render(text)
    if page == "data":
        ctx["files"] = [
            {
                "name": f.name,
                "href": f"{ctx['prefix']}data/{f.name}",
                "content": f.content[lang],
                "rows": num(f.rows, 0, lang) if f.rows else "",
                "size": _size(f.bytes, lang),
            }
            for f in files
        ]
        from prognosebuch.site.exports import DICTIONARY

        ctx["dictionary"] = [
            (tbl, col, typ, unit, de if lang == "de" else en)
            for tbl, col, typ, unit, en, de in DICTIONARY
        ]
    return ctx


def _size(n: int, lang: str) -> str:
    if n >= 1_000_000:
        return f"{num(n / 1_000_000, 1, lang)} MB"
    return f"{num(max(n / 1000, 1), 0, lang)} kB"


TEMPLATES = {
    "home": "home.html",
    "record": "record.html",
    "failures": "failures.html",
    "cheapest": "cheapest.html",
    "method": "method.html",
    "data": "data.html",
}


def build_site(root: Path, out: Path, now_utc: datetime) -> list[Path]:
    sd = load_site_data(root, now_utc)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(HERE / "static", out / "static")
    files = build_exports(sd, out / "data")
    env = _env()
    written = []
    for lang in LANGS:
        for page, tpl in TEMPLATES.items():
            ctx = page_context(sd, page, lang, files)
            html = env.get_template(tpl).render(**ctx)
            path = out / PAGES[page][lang] / "index.html"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html)
            written.append(path)
    (out / ".nojekyll").write_text("")
    return written
