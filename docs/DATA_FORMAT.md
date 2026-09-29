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

The manifest also holds `cheapest_windows` (from 2026-09-30): for each target day the
3-hour window with the lowest mean median forecast (`start_utc`, `end_utc`, `start_local`,
`end_local`, `expected_mean_eur_mwh`, `day_mean_eur_mwh`), the probability that it is the
cheapest window (`p_cheapest`) and that it is within 5 EUR/MWh of the cheapest (`p_near`),
the number of error scenarios used, and the three most likely cheapest windows. Method:
`src/prognosebuch/cheapest.py`.

## `actuals/day_ahead_price_de_lu/YYYY-MM.parquet`

Day-ahead prices from SMARD.de (Bundesnetzagentur | SMARD.de, CC BY 4.0). May be updated
if SMARD revises a value.

| Column | Type | Description |
|---|---|---|
| delivery_start_utc | timestamp (UTC) | Start of the quarter-hour |
| delivery_start_local | string | Europe/Berlin, ISO 8601 with offset |
| price_eur_mwh | float64 | Day-ahead price DE-LU |

## `inputs/weather/open_meteo_icon/YYYY-MM.parquet`

Weather forecast inputs (Open-Meteo Previous Runs, ICON; "Weather data by Open-Meteo.com",
CC BY 4.0), one row per valid hour (UTC) and lead time: `valid_utc`, `lead_days` (2 or 3),
`wind_power` (mean normalised turbine output 0–1 over wind points), `wind_speed` (m/s at 100 m),
`solar` (mean shortwave radiation in W/m² for the hour starting at `valid_utc`), `temp` (°C at
2 m), `available_at_utc` (valid time − lead + 6 h, used for the information cutoff).

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

## `scores/cheapest/YYYY/MM/<target_date>.parquet`

One row per model version and horizon that was due: `status` (`scored`/`missed`), the
recommended window (`window_start_utc`, `window_start_local`, `expected_mean_eur_mwh`,
`p_cheapest`, `p_near`), and how it turned out: `actual_best_start_utc`,
`actual_mean_recommended`, `actual_mean_best`, `actual_day_mean`, `hit` (was the cheapest),
`near` (within 5 EUR/MWh), `regret_eur_mwh`. Summarised in `scores/summary.json` under
`cheapest_windows` (hit rate vs. stated probability, Brier score, regret, saving vs. day mean).

## `backtest/<first>_<last>/`

Rolling backtest output (not part of the book): `summary.json` (same structure as
`scores/summary.json`, plus `kind: backtest` and a warning) and `daily.csv` (per model version,
horizon and target day: `mae`, `rmse`, `pinball`, `coverage_80`, `n_slots`).

## `scores/summary.json`

Regenerated on every evaluation: rolling windows (7, 30, 90 days, all) per model and
horizon with MAE, RMSE, pinball, 80 % band coverage and MAE skill versus the reference
model (`naive_similar_day.v1`), split into all / weekday / weekend / negative prices /
price spikes; pairwise Diebold-Mariano tests; list of all missed forecasts.
