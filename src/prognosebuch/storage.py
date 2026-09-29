"""File layout and schemas of the public data.

Layout (all paths relative to the repository root):

    forecasts/YYYY/MM/<issue_date>/<model>.v<version>.parquet   write-once forecast
    forecasts/YYYY/MM/<issue_date>/<model>.v<version>.json      write-once manifest
    actuals/day_ahead_price_de_lu/YYYY-MM.parquet               observed prices (may be revised)
    scores/YYYY/MM/<target_date>.parquet                        per-quarter-hour evaluation
    scores/summary.json                                         rolling metrics (regenerated)

See docs/DATA_FORMAT.md for column descriptions.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FORECAST_SCHEMA_VERSION = 1
SCORE_SCHEMA_VERSION = 1

FORECAST_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int16()),
        ("issue_date", pa.date32()),
        ("issued_at_utc", pa.timestamp("s", tz="UTC")),
        ("model", pa.string()),
        ("model_version", pa.string()),
        ("horizon_days", pa.int8()),
        ("target_date", pa.date32()),
        ("delivery_start_utc", pa.timestamp("s", tz="UTC")),
        ("delivery_start_local", pa.string()),
        ("resolution", pa.string()),
        ("q10", pa.float64()),
        ("q50", pa.float64()),
        ("q90", pa.float64()),
    ]
)

ACTUALS_SCHEMA = pa.schema(
    [
        ("delivery_start_utc", pa.timestamp("s", tz="UTC")),
        ("delivery_start_local", pa.string()),
        ("price_eur_mwh", pa.float64()),
    ]
)

CHEAPEST_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int16()),
        ("target_date", pa.date32()),
        ("model", pa.string()),
        ("model_version", pa.string()),
        ("horizon_days", pa.int8()),
        ("issue_date", pa.date32()),
        ("status", pa.string()),
        ("window_start_utc", pa.timestamp("s", tz="UTC")),
        ("window_start_local", pa.string()),
        ("expected_mean_eur_mwh", pa.float64()),
        ("p_cheapest", pa.float64()),
        ("p_near", pa.float64()),
        ("actual_best_start_utc", pa.timestamp("s", tz="UTC")),
        ("actual_mean_recommended", pa.float64()),
        ("actual_mean_best", pa.float64()),
        ("actual_day_mean", pa.float64()),
        ("hit", pa.bool_()),
        ("near", pa.bool_()),
        ("regret_eur_mwh", pa.float64()),
        ("scored_at_utc", pa.timestamp("s", tz="UTC")),
    ]
)

SCORE_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int16()),
        ("target_date", pa.date32()),
        ("delivery_start_utc", pa.timestamp("s", tz="UTC")),
        ("delivery_start_local", pa.string()),
        ("model", pa.string()),
        ("model_version", pa.string()),
        ("horizon_days", pa.int8()),
        ("issue_date", pa.date32()),
        ("status", pa.string()),
        ("actual", pa.float64()),
        ("q10", pa.float64()),
        ("q50", pa.float64()),
        ("q90", pa.float64()),
        ("error", pa.float64()),
        ("abs_error", pa.float64()),
        ("sq_error", pa.float64()),
        ("pinball", pa.float64()),
        ("in_band", pa.bool_()),
        ("scored_at_utc", pa.timestamp("s", tz="UTC")),
    ]
)


def forecast_dir(root: Path, issue_date: date) -> Path:
    return root / "forecasts" / f"{issue_date:%Y}" / f"{issue_date:%m}" / issue_date.isoformat()


def forecast_paths(root: Path, issue_date: date, model: str, version: str) -> tuple[Path, Path]:
    d = forecast_dir(root, issue_date)
    stem = f"{model}.v{version}"
    return d / f"{stem}.parquet", d / f"{stem}.json"


def actuals_path(root: Path, month: str) -> Path:
    return root / "actuals" / "day_ahead_price_de_lu" / f"{month}.parquet"


def score_path(root: Path, target_date: date) -> Path:
    return root / "scores" / f"{target_date:%Y}" / f"{target_date:%m}" / f"{target_date}.parquet"


def cheapest_score_path(root: Path, target_date: date) -> Path:
    d = root / "scores" / "cheapest" / f"{target_date:%Y}" / f"{target_date:%m}"
    return d / f"{target_date}.parquet"


def to_parquet_bytes(df: pd.DataFrame, schema: pa.Schema) -> bytes:
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=True)
    table = table.replace_schema_metadata(None)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()


def read_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_once(path: Path, data: bytes) -> None:
    """Create ``path`` with ``data``; fail if it already exists (forecasts are immutable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as fh:
        fh.write(data)


def dump_json(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False) + "\n").encode()
