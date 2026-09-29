"""Regenerate tests/fixtures/smard_prices_sample.csv from the live SMARD API.

Two windows of real quarter-hourly DE-LU day-ahead prices: October/November 2025
(autumn DST switch on 2025-10-26, 100 quarter-hours) and March/April 2026 (spring DST
switch on 2026-03-29, 92 quarter-hours; negative prices around Easter).
Source: Bundesnetzagentur | SMARD.de, CC BY 4.0.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from prognosebuch.smard import PRICE_DE_LU, SmardClient
from prognosebuch.timeutil import day_bounds_utc, local_iso

WINDOWS = [(date(2025, 10, 1), date(2025, 11, 9)), (date(2026, 3, 10), date(2026, 4, 12))]
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "smard_prices_sample.csv"


def main() -> None:
    client = SmardClient()
    parts = []
    for first, last in WINDOWS:
        s = client.fetch(PRICE_DE_LU, day_bounds_utc(first)[0], day_bounds_utc(last)[1])
        parts.append(s)
    s = pd.concat(parts)
    idx = pd.DatetimeIndex(s.index)
    df = pd.DataFrame(
        {
            "delivery_start_utc": idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "delivery_start_local": local_iso(idx),
            "price_eur_mwh": s.to_numpy(),
        }
    )
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows to {OUT}")


if __name__ == "__main__":
    main()
