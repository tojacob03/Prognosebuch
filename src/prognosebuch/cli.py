"""Command line entry point: ``prognosebuch forecast|evaluate|audit|probe``.

Exit codes: 0 success or nothing to do, 1 a due forecast could not be issued in time or a
check failed, 2 partial failure (some models written, some failed).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from prognosebuch.audit import run_audit
from prognosebuch.evaluate import run_evaluate
from prognosebuch.forecast import (
    DeadlinePassedError,
    PricesAlreadyPublishedError,
    RunInfo,
    TooEarlyError,
    run_forecast,
)
from prognosebuch.probe import run_probe
from prognosebuch.smard import SmardClient
from prognosebuch.timeutil import utc_now


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
    args = p.parse_args(argv)
    rc: int = args.func(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
