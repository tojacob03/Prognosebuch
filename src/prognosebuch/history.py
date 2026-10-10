"""Archive of published day-ahead prices (``actuals/``) and the price history for models.

SMARD's quarter-hour price series reaches back to October 2018. Before the switch to a
15-minute day-ahead auction on 2025-10-01 it holds the hourly price repeated for each
quarter-hour, so hourly means are consistent across the whole history.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from prognosebuch import energycharts
from prognosebuch.smard import PRICE_DE_LU, JsonGetter, SmardClient, http_get_json
from prognosebuch.storage import ACTUALS_SCHEMA, actuals_path, read_parquet, to_parquet_bytes
from prognosebuch.timeutil import (
    TZ,
    day_bounds_utc,
    day_slots_utc,
    local_date,
    local_iso,
    normalize_index,
)

ACTUALS_DIR = Path("actuals") / "day_ahead_price_de_lu"


def _empty() -> pd.Series:
    return pd.Series(
        [], index=pd.DatetimeIndex([], tz="UTC", name="delivery_start_utc"), dtype="float64"
    )


def load_actuals_file(path: Path) -> pd.Series:
    if not path.exists():
        return _empty()
    df = read_parquet(path)
    s = pd.Series(
        df["price_eur_mwh"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(df["delivery_start_utc"], name="delivery_start_utc"),
    )
    return normalize_index(s)


def load_actuals(root: Path) -> pd.Series:
    files = sorted((root / ACTUALS_DIR).glob("*.parquet"))
    parts = [load_actuals_file(f) for f in files]
    if not parts:
        return _empty()
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="last")]


def merge_into_archive(root: Path, fresh: pd.Series) -> list[Path]:
    """Merge prices into the monthly archive files (newer values win); return changed files."""
    if fresh.empty:
        return []
    months = pd.Series(
        pd.DatetimeIndex(fresh.index).tz_convert(TZ).strftime("%Y-%m"), index=fresh.index
    )
    changed: list[Path] = []
    for month, idx in fresh.groupby(months).groups.items():
        path = actuals_path(root, str(month))
        old = load_actuals_file(path)
        merged = pd.concat([old, fresh.loc[idx]])
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


FALLBACK_LOG = ACTUALS_DIR / "fallback_log.csv"


def update_actuals(
    root: Path,
    client: SmardClient,
    now_utc: datetime,
    lookback_days: int = 21,
    fallback_get_json: JsonGetter | None = http_get_json,
) -> list[Path]:
    """Merge recent SMARD prices; fill quarter-hours SMARD lacks from Energy-Charts."""
    today = local_date(pd.Timestamp(now_utc))
    first = today - timedelta(days=lookback_days)
    start = day_bounds_utc(first)[0]
    end = day_bounds_utc(today + timedelta(days=2))[1]
    changed = merge_into_archive(root, client.fetch(PRICE_DE_LU, start, end))
    if fallback_get_json is None:
        return changed
    archive = load_actuals(root)
    last = max(archive.index.max().tz_convert(TZ).date(), today) if len(archive) else today
    _, gap = energycharts.fill_gaps(archive, first, last, fallback_get_json)
    if gap.error:
        print(f"::warning::Energy-Charts fallback failed: {gap.error}")
    if not gap.filled.empty:
        changed += merge_into_archive(root, gap.filled)
        log_fallback(root, gap, pd.Timestamp(now_utc).floor("s"))
    return changed


def log_fallback(root: Path, gap: energycharts.GapFill, fetched_at: pd.Timestamp) -> Path:
    """Append which delivery days were completed from Energy-Charts (and how many values)."""
    path = root / FALLBACK_LOG
    local = pd.DatetimeIndex(gap.filled.index).tz_convert(TZ)
    counts = pd.Series(1, index=local.date).groupby(level=0).sum()
    rows = pd.DataFrame(
        {
            "fetched_at_utc": fetched_at.isoformat(),
            "delivery_date_local": [d.isoformat() for d in counts.index],
            "quarter_hours_filled": counts.to_numpy(),
            "source": energycharts.ATTRIBUTION,
        }
    )
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path


def backfill_actuals(root: Path, client: SmardClient, since: date, until: date) -> list[Path]:
    start, _ = day_bounds_utc(since)
    _, end = day_bounds_utc(until)
    return merge_into_archive(root, client.fetch(PRICE_DE_LU, start, end))


def day_is_complete(prices: pd.Series, d: date) -> bool:
    return bool(prices.reindex(day_slots_utc(d)).notna().all())


def price_history(archive: pd.Series, fresh: pd.Series) -> pd.Series:
    """Archive plus freshly fetched prices; fresh values win."""
    s = pd.concat([normalize_index(archive), normalize_index(fresh)]).sort_index()
    return s[~s.index.duplicated(keep="last")]
