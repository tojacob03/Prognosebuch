# Data format (schema version 1)

All timestamps are stored in UTC. A *delivery day* is a calendar day in Europe/Berlin
(96 quarter-hours; 92 on the spring and 100 on the autumn DST switch). Prices are in
EUR/MWh. Resolution is 15 minutes (`PT15M`); hourly values are the mean of the four
quarter-hours.

Schema changes are versioned (`schema_version`) and backwards compatible: existing
files are never rewritten; new columns may be added in a new schema version.

## `forecasts/YYYY/MM/<issue_date>/<model>.v<version>.parquet`

Write-once. One file per model version and issue day, containing the D+1 and D+2
forecast. CI fails if any file under `forecasts/` is ever modified or deleted.

| Column | Type | Description |
|---|---|---|
| schema_version | int16 | Format version (1) |
| issue_date | date | Day D (Europe/Berlin) on which the forecast was issued |
| issued_at_utc | timestamp (UTC) | Exact issue time |
| model | string | Model name, e.g. `naive_weekly` |
| model_version | string | Model version; logic changes create a new version |
| horizon_days | int8 | 1 = D+1, 2 = D+2 |
| target_date | date | Delivery day (Europe/Berlin) |
| delivery_start_utc | timestamp (UTC) | Start of the quarter-hour |
| delivery_start_local | string | Same instant in Europe/Berlin, ISO 8601 with offset |
| resolution | string | `PT15M` |
| q10, q50, q90 | float64 | 10 %, 50 % (median, point forecast) and 90 % quantile |

## `forecasts/.../<model>.v<version>.json` (manifest)

Write-once. `issued_at_utc`, `issued_at_local`, `target_dates`, `data_cutoffs` (per
source: fetch time, information cutoff, last value used, last value available),
`package_version`, `code_commit`, `workflow_run_url` and `parquet_sha256` (SHA-256 of
the parquet file, checked by `prognosebuch audit`).

## `actuals/day_ahead_price_de_lu/YYYY-MM.parquet`

Day-ahead prices from SMARD.de (Bundesnetzagentur | SMARD.de, CC BY 4.0). May be updated
if SMARD revises a value.

| Column | Type | Description |
|---|---|---|
| delivery_start_utc | timestamp (UTC) | Start of the quarter-hour |
| delivery_start_local | string | Europe/Berlin, ISO 8601 with offset |
| price_eur_mwh | float64 | Day-ahead price DE-LU |

## `scores/YYYY/MM/<target_date>.parquet`

One row per model version, horizon and quarter-hour of the target day that was *due*.
Missing forecasts are kept as rows with `status = missed`.

| Column | Type | Description |
|---|---|---|
| schema_version | int16 | 1 |
| target_date, delivery_start_utc, delivery_start_local | | as above |
| model, model_version, horizon_days, issue_date | | which forecast is scored |
| status | string | `scored`, `missed` (no forecast file) or `incomplete` |
| actual | float64 | Published price |
| q10, q50, q90 | float64 | Forecast (null if missed) |
| error | float64 | actual − q50 |
| abs_error, sq_error | float64 | \|error\|, error² |
| pinball | float64 | Mean pinball loss over q10/q50/q90 |
| in_band | bool | actual within [q10, q90] |
| scored_at_utc | timestamp (UTC) | When the row was computed |

## `scores/summary.json`

Regenerated on every evaluation: rolling windows (7, 30, 90 days, all) per model and
horizon with MAE, RMSE, pinball, 80 % band coverage and MAE skill versus the reference
model (`naive_similar_day.v1`), split into all / weekday / weekend / negative prices /
price spikes; pairwise Diebold-Mariano tests; list of all missed forecasts.
