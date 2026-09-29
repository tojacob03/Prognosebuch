"""Command line entry point: ``prognosebuch forecast|evaluate|audit|probe``.

Exit codes: 0 success or nothing to do, 1 a due forecast could not be issued in time or a
check failed, 2 partial failure (some models written, some failed).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

from prognosebuch.audit import run_audit
from prognosebuch.backtest import run_backtest, write_backtest
from prognosebuch.evaluate import run_evaluate
from prognosebuch.forecast import (
    DeadlinePassedError,
    PricesAlreadyPublishedError,
    RunInfo,
    TooEarlyError,
    run_forecast,
)
from prognosebuch.history import backfill_actuals, load_actuals
from prognosebuch.probe import run_probe
from prognosebuch.registry import CATALOG
from prognosebuch.smard import SmardClient
from prognosebuch.timeutil import utc_now
from prognosebuch.weather import load_weather


def _now(value: str | None) -> datetime:
    if value is None:
        return utc_now()
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise SystemExit("--now needs an explicit UTC offset, e.g. 2026-09-30T07:05:00+00:00")
    return dt


def cmd_forecast(args: argparse.Namespace) -> int:
    root = Path(args.root)
    try:
        res = run_forecast(root, utc_now(), SmardClient(), RunInfo.from_env())
    except TooEarlyError as exc:
        print(f"skip: {exc}")
        return 0
    except (DeadlinePassedError, PricesAlreadyPublishedError) as exc:
        print(f"::error::forecast not issued: {exc}")
        return 1
    for key in res.skipped_existing:
        print(f"already issued for {res.issue_date}: {key}")
    for p in res.written:
        print(f"wrote {p.relative_to(root)}")
    for key, msg in res.failed.items():
        print(f"::error::{key} failed: {msg}")
    return 2 if res.failed else 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    root = Path(args.root)
    client = None if args.offline else SmardClient()
    res = run_evaluate(root, _now(args.now), client)
    for p in res.actuals_changed:
        print(f"updated {p.relative_to(root)}")
    for d in res.scored_days:
        print(f"scored target day {d}")
    if res.pending_days:
        print("pending (prices not complete yet): " + ", ".join(map(str, res.pending_days)))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    problems = run_audit(Path(args.root))
    for p in problems:
        print(f"::error::{p}")
    print("audit ok" if not problems else f"audit failed: {len(problems)} problem(s)")
    return 1 if problems else 0


def cmd_probe(args: argparse.Namespace) -> int:
    n = run_probe(Path(args.out), SmardClient(), _now(args.now))
    print(f"probed {n} series")
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    root = Path(args.root)
    changed = backfill_actuals(
        root, SmardClient(), date.fromisoformat(args.since), date.fromisoformat(args.until)
    )
    print(f"updated {len(changed)} archive file(s)")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    root = Path(args.root)
    keys = args.models.split(",") if args.models else list(CATALOG)
    unknown = [k for k in keys if k not in CATALOG]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}; known: {list(CATALOG)}")
    first, last = date.fromisoformat(args.first), date.fromisoformat(args.last)
    res = run_backtest(
        [CATALOG[k] for k in keys],
        load_actuals(root),
        first,
        last,
        utc_now(),
        args.jobs,
        weather=load_weather(root),
    )
    out = Path(args.out) if args.out else root / "backtest" / f"{first}_{last}"
    for p in write_backtest(res, out):
        print(f"wrote {p}")
    return 0


def cmd_site(args: argparse.Namespace) -> int:
    from prognosebuch.site.build import build_site

    written = build_site(Path(args.root), Path(args.out), utc_now())
    print(f"wrote {len(written)} pages to {args.out}")
    return 0


def cmd_release_notes(args: argparse.Namespace) -> int:
    root = Path(args.root)
    files = sorted((root / "forecasts").rglob("*.parquet"))
    issues = sorted({f.parent.name for f in files})
    lines = [
        "Weekly snapshot of the Prognosebuch book: every forecast file as committed before the",
        "auction, the scores and the actual prices, plus SHA-256 checksums of all forecast files.",
        "",
        f"- Forecast files: {len(files)}",
        f"- Issue days: {len(issues)}" + (f" ({issues[0]} to {issues[-1]})" if issues else ""),
    ]
    summary_path = root / "scores" / "summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        lines += [f"- Missed forecasts (model x horizon x day): {len(s.get('missed', []))}", ""]
        lines += [
            "| Model | Horizon | Days due / forecast | MAE | Skill vs. reference |",
            "|---|---|---|---|---|",
        ]
        for r in s["windows"]["all"]["models"]:
            a = r["all"]
            lines.append(
                f"| {r['model']} | D+{r['horizon_days']} | {r['days_due']} / {r['days_forecast']} "
                f"| {a['mae']} | {a['skill_mae']} |"
            )
    lines += ["", "Data: CC BY 4.0. Prices: Bundesnetzagentur | SMARD.de. Not investment advice."]
    print("\n".join(lines))
    return 0


def cmd_headline(args: argparse.Namespace) -> int:
    """Print the live numbers for README, case study and posts (never backtest numbers)."""
    root = Path(args.root)
    path = root / "scores" / "summary.json"
    if not path.exists():
        print("No live scores yet.")
        return 0
    s = json.loads(path.read_text())
    w = s["windows"]["all"]
    print(f"Live period: {w['from']} to {w['to']} (target days)")
    print(f"Missed forecasts (model x horizon x day): {len(s.get('missed', []))}")
    for r in sorted(w["models"], key=lambda r: (r["horizon_days"], r["all"]["mae"] or 1e9)):
        a = r["all"]
        print(
            f"  D+{r['horizon_days']} {r['model']:<22} days {r['days_forecast']}/{r['days_due']}"
            f"  MAE {a['mae']}  RMSE {a['rmse']}  skill {a['skill_mae']}"
            f"  pinball {a['pinball']}  coverage80 {a['coverage_80']}"
        )
    for d in w.get("dm_tests", []):
        if "p_value" in d and s["reference_model"] in (d["model_a"], d["model_b"]):
            print(
                f"  DM D+{d['horizon_days']} {d['model_a']} vs {d['model_b']}: "
                f"stat {d['statistic']}, p {d['p_value']}, n {d['n_days']}"
            )
    for r in s.get("cheapest_windows", {}).get("windows", {}).get("all", []):
        if r.get("days_scored"):
            print(
                f"  cheapest 3h D+{r['horizon_days']} {r['model']}: hit {r['hit_rate']} "
                f"(stated {r['mean_p_cheapest']}), within 0.5 ct {r['near_rate']} "
                f"(stated {r['mean_p_near']}), saving vs day mean "
                f"{r['saving_vs_day_mean_eur_mwh']} EUR/MWh"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="prognosebuch")
    p.add_argument("--root", default=".", help="repository root (default: .)")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forecast", help="issue today's forecasts for D+1 and D+2")
    # Deliberately no --now override: the issue time is always the real clock.
    f.set_defaults(func=cmd_forecast)
    e = sub.add_parser("evaluate", help="update actuals, score due days, write summary")
    e.add_argument("--now")
    e.add_argument("--offline", action="store_true", help="do not fetch new prices")
    e.set_defaults(func=cmd_evaluate)
    a = sub.add_parser("audit", help="verify immutability and manifests of all forecasts")
    a.set_defaults(func=cmd_audit)
    pr = sub.add_parser("probe", help="log how far each SMARD series reaches right now")
    pr.add_argument("--out", required=True)
    pr.add_argument("--now")
    pr.set_defaults(func=cmd_probe)
    b = sub.add_parser("backfill", help="load past prices from SMARD into actuals/")
    b.add_argument("--since", required=True)
    b.add_argument("--until", required=True)
    b.set_defaults(func=cmd_backfill)
    bt = sub.add_parser("backtest", help="rolling-origin backtest on the price archive")
    bt.add_argument("--first", required=True, help="first target date")
    bt.add_argument("--last", required=True, help="last target date")
    bt.add_argument("--models", help="comma-separated model keys (default: all)")
    bt.add_argument("--out", help="output directory (default: backtest/<first>_<last>)")
    bt.add_argument("--jobs", type=int, default=-1, help="parallel workers (default: all cores)")
    bt.set_defaults(func=cmd_backtest)
    st = sub.add_parser("site", help="build the static website and data exports")
    st.add_argument("--out", default="_site")
    st.set_defaults(func=cmd_site)
    hl = sub.add_parser("headline", help="print live headline numbers (for README and posts)")
    hl.set_defaults(func=cmd_headline)
    rn = sub.add_parser("release-notes", help="print markdown notes for the weekly release")
    rn.set_defaults(func=cmd_release_notes)
    args = p.parse_args(argv)
    rc: int = args.func(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
