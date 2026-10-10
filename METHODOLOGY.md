# Methodology

This document describes what is forecast, when, with which information, how forecasts are
scored, and what the numbers can and cannot tell you.

**Not investment advice, not a trading signal, no promise of savings.**

## 1. Target

The day-ahead price of the German-Luxembourg bidding zone (DE-LU) from the EPEX SPOT day-ahead
auction (SDAC), as published by the Bundesnetzagentur on SMARD.de. Since 2025-10-01 the auction
clears in 15-minute products; all forecasts and scores are per quarter-hour. Hourly values are
the mean of the four quarter-hours.

A *delivery day* is a calendar day in Europe/Berlin: 96 quarter-hours, 92 on the last Sunday of
March, 100 on the last Sunday of October.

## 2. Timeline and the information cutoff

| Europe/Berlin | Event |
|---|---|
| D-1, ~12:45 | Prices for day D are published |
| **D, 08:45–12:00** | **Forecast window. Prognosebuch issues forecasts for D+1 and D+2 (target ~09:00)** |
| D, 12:00 | Auction gate closure for D+1 — hard deadline |
| D, ~12:45 | Prices for D+1 are published; the D+1 forecast can be scored |
| D+1, ~12:45 | Prices for D+2 are published; the D+2 forecast can be scored |

**Issue time** is the moment the forecast file was produced (`issued_at_utc` in the manifest).
**Information cutoff** is the latest point in time whose data a model may use. For prices it is
the end of day D: all quarter-hours of D are known at issue time because they were published the
day before, and nothing of D+1 is known.

Guards, all enforced in code and tested:

- The job only issues inside the window; after 12:00 it refuses and the day is recorded as missed.
- If SMARD already shows any price for D+1, the job refuses.
- If SMARD lacks quarter-hours, they are filled from Energy-Charts, which republishes the same
  SMARD series (identical on every quarter-hour both have); the manifest records it.
- Models receive an `InfoSet` that hard-cuts all prices after the cutoff. A test sets every later
  price to an absurd value and checks that every model's forecast is bit-for-bit unchanged.
- The LEAR model additionally cuts the data itself, and its features only use days `t-h` and
  earlier.

GitHub's cron scheduler proved unreliable for this repository (runs up to six hours late or
dropped; the first issue day was missed because of it, see `INCIDENTS.md`). A `clock` workflow
therefore waits for the timetable in `src/prognosebuch/clock.py` and dispatches the forecast job
at 08:55, 09:30 and 10:30 Berlin time; it hands over to a fresh run of itself before GitHub's
six-hour job limit. Cron entries remain as a backup. The job checks local time itself; only the
first successful run of a day writes, later runs only add models that are still missing.

## 3. The book (live track record)

- Every forecast is written once as `forecasts/YYYY/MM/<issue_date>/<model>.v<version>.parquet`
  plus a JSON manifest (issue time, per-source data cutoffs, code commit, workflow run, SHA-256).
- Files are opened in exclusive-create mode; the bot commits additions only; `prognosebuch audit`
  (run in CI on every push, daily, and after every job) fails if any forecast file was ever modified,
  deleted, or if a manifest hash or issue time does not match.
- The `evaluate` job scores every target day once its prices are complete and the last forecast
  deadline for it has passed. Every forecast that was due but does not exist is written as a row
  with `status = missed`. Missed days are counted and listed, never dropped.
- A model is identified by name and version. Any change to a model's logic creates a new version
  that runs in parallel; the record of all versions stays visible.

What an outsider can verify: file hashes, the git history of `forecasts/` (additions only), the
Actions run linked in each manifest (with its log and timing), and — once set up — weekly
releases archived with a DOI. Git commit timestamps alone are not proof; the push time on GitHub
and the workflow logs are.

## 4. Models

| Model | Idea |
|---|---|
| `naive_last_day.v1` | Copy day D onto the target day |
| `naive_weekly.v1` | Copy the same weekday one week before the target day |
| `naive_similar_day.v1` | **Reference.** Standard benchmark from Lago et al. (2021): Mon/Sat/Sun from one week before, Tue–Fri from the day before (for D+2 the last known day) |
| `lear.v1` | Lasso-estimated autoregression (see below) |
| `gbm.v1` | Gradient boosting with quantile loss on weather forecasts, lagged prices and calendar (see below) |

Copying uses local wall-clock time, so DST days work (a repeated hour takes the single hour's
value; a doubled source hour is averaged; a missing hour is filled from the previous one).

**LEAR** (independent implementation after Lago, Marcjasz, De Schutter & Weron, 2021):

- One linear model per delivery hour and horizon on hourly mean prices.
- Inputs: the 24 hourly prices of days `t-h`, `t-h-1`, `t-h-2` and `t-7`; day-of-week dummies;
  a German public holiday flag.
- Transform: `asinh((x - median) / MAD)`, per column, estimated on the calibration window.
- L1 penalty per hour chosen by the Akaike information criterion on the LARS path.
- Average of three calibration windows: 182, 364 and 728 days, re-estimated every day.
- Quarter-hours: hourly forecast plus the mean intra-hour profile of the last 28 days.

**Gradient boosting** (`gbm.v1`): scikit-learn's histogram gradient boosting (the algorithm
family of LightGBM, chosen because it needs no native OpenMP library) with quantile loss, one
model per quantile (P10, P50, P90) and horizon, pooled over the 24 hours.

- Weather: Open-Meteo Previous Runs of the DWD ICON model at 12 points (wind at 100 m for
  onshore/offshore wind areas, solar radiation, temperature), aggregated to a wind power proxy,
  mean wind speed, mean irradiance and mean temperature. D+1 uses values forecast 48 h before the
  valid time, D+2 72 h. A value counts as available 6 h after the run it comes from; the
  information set rejects anything that was not available at 08:45 on the issue day.
- Also: daily means of the target day and of the last known day and their difference; lagged
  prices (last known day, one day before, one week before; mean/min/max of the last known day;
  seven-day mean); hour, weekday, weekend, holiday, season.
- The target is the price minus the seven-day mean, so trees learn shape and weather effects
  while the level comes from recent prices.
- Re-estimated weekly on all days up to the most recent Sunday; training starts in March 2024
  (the Previous Runs archive begins in February 2024).
- Its P10/P90 come from the quantile models themselves; quantiles are sorted if they cross.

**Uncertainty bands (naive models and LEAR).** The 10 % and 90 % quantiles are the point forecast plus the
10 %/90 % quantiles of the model's own errors on the previous 60 (LEAR) or 90 (naive) target days,
per local hour and horizon, where each past forecast is recomputed exactly as it would have been
issued on its own issue date. This is simple and honest, but it assumes the recent error
distribution holds tomorrow; it is not conditional on, e.g., a windy day.

## 5. Metrics

Per model version, horizon (D+1, D+2) and window (last 7, 30, 90 days, all):

| Metric | Definition |
|---|---|
| MAE | mean \|actual − q50\| in EUR/MWh |
| RMSE | root mean squared error |
| Skill (MAE) | 1 − MAE(model) / MAE(reference) on the same quarter-hours; > 0 = better than the rule of thumb |
| Pinball loss | mean quantile loss over q10, q50, q90 (lower is better; rewards sharp *and* calibrated bands) |
| Coverage 80 % | share of actual prices inside [q10, q90]; should be about 0.80 |
| Days due / forecast | how many target days were due and how many had a forecast |

Segments: all quarter-hours, weekdays, weekends, negative actual prices, price spikes (actual at or
above the 90th percentile of the window).

**Diebold-Mariano test** (with the Harvey-Leybourne-Newbold small-sample correction) on the series
of daily MAEs of two models, paired by target day, two-sided. Reported from 10 paired days on;
with few days the test has little power and "no significant difference" is the expected result.

## 6. Cheapest 3 hours

For households on dynamic tariffs the forecast is turned into one recommendation per day: the
3-hour window (12 consecutive quarter-hours within the delivery day) with the lowest mean
median forecast. Its probabilities come from scenarios: the median forecast plus each of the
model's own error curves from the previous 60–90 days (the same curves the bands are built from),
aligned by local clock time, so the within-day shape of real errors is kept.

- `p_cheapest`: share of scenarios in which the recommended window is the cheapest window.
- `p_near`: share of scenarios in which it costs at most 5 EUR/MWh (0.5 ct/kWh) more than the
  cheapest window.

Both are written into the forecast manifest before the auction and scored afterwards: hit rate
and "near" rate against the stated probabilities (calibration), Brier score, the extra cost versus
the best window, and the saving versus the day mean. The recommendation uses exchange prices;
taxes, levies and grid fees are added per kWh and do not change the order of the hours.

## 7. Backtest vs. live

The backtest (`prognosebuch backtest`) re-runs the same code for every past day with only the data
known then; a test checks that it reproduces the live forecast exactly. It is still **not** a test
of the future: the model design was chosen by someone who had seen the past. Backtest numbers are
shown separately and labelled as such. Only the live book counts as evidence.

## 8. Limitations

- **No gas or CO₂ prices.** They drive the price level but no openly licensed source exists.
  Level shifts caused by fuel prices are learned only through lagged prices.
- **No grid-operator load or renewable forecasts yet.** SMARD does not keep vintages, so it is
  unknown what was available at 09:00 on past days. An hourly probe is measuring this; such inputs
  are only added once they are demonstrably available before issue time.
- Weather forecasts come from the Previous Runs API with a lead time long enough that the data
  existed at issue time (D+1: 48 h, D+2: 72 h), not from the Historical Forecast API, which is
  close to observed weather and would leak information. The price of this caution: live, a
  fresher weather run would be available; the model deliberately does not use it, so that
  training and live use the same lead time.
- Bands are unconditional (see above) and quarter-hour errors within a day are strongly
  correlated; a day with a bad forecast is usually bad for many hours at once.
- Rare events (extreme spikes, outages, market coupling incidents) are not predictable from these
  inputs.
