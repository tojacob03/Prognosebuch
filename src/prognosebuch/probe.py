"""Availability probe: log how far each SMARD series reaches at the time of probing.

SMARD keeps no vintages, so the only way to learn when, e.g., the grid operators' wind
forecast for D+1 becomes available is to look repeatedly and write it down. Features
for later models are only allowed if this log shows they exist before the issue time.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

from prognosebuch.smard import PROBE_SERIES, SmardClient

FIELDS = [
    "probed_at_utc",
    "series",
    "description",
    "last_value_start_utc",
    "last_value_start_local",
]


def run_probe(out: Path, client: SmardClient, now_utc: datetime) -> int:
    probed = pd.Timestamp(now_utc).floor("s").isoformat()
    new_file = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for s in PROBE_SERIES:
            try:
                last = client.last_timestamp(s)
            except RuntimeError:
                last = None
            w.writerow(
                {
                    "probed_at_utc": probed,
                    "series": s.key,
                    "description": s.description,
                    "last_value_start_utc": last.isoformat() if last is not None else "",
                    "last_value_start_local": last.tz_convert("Europe/Berlin").isoformat()
                    if last is not None
                    else "",
                }
            )
    return len(PROBE_SERIES)
