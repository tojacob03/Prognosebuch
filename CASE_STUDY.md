# Case study: a forecast you can check

*Status: the live track record starts on 2026-09-30. Section 3 reports only backtest numbers,
which are labelled as such; the live numbers are added once at least four weeks exist
(`uv run prognosebuch headline` prints them from the committed scores).*

## 1. Problem

German day-ahead electricity prices are set every day at 12:00 for the next day, in 96
quarter-hours. They matter for households on dynamic tariffs, for batteries, heat pumps and
electric cars, and for anyone planning flexible load. Price forecasts are easy to find. What is
hard to find is evidence of how good they are: most published forecasts show a nice chart and no
record, and most open-source projects report a backtest, which is always tuned on the past it is
tested on.

Question: **How well can DE-LU day-ahead prices for tomorrow and the day after be forecast, and
how reliable is a model compared with simple rules of thumb, measured on real days?**

The answer has to be checkable by someone who does not trust the author.

## 2. Approach

**Make cheating hard, then measure.**

- *Pre-registration.* Every morning (target 09:00 Berlin, hard deadline 12:00 = auction gate
  closure) a scheduled GitHub Actions job writes the forecast for D+1 and D+2 as a new,
  write-once Parquet file plus a JSON manifest: issue time, the last data point of every
  source, code commit, workflow run and SHA-256. CI fails if any existing forecast file is ever
  modified or deleted; the job commits additions only; `main` is protected against force pushes.
- *Automatic scoring.* After the prices are published, a second job scores every forecast that
  was due. A missing forecast is recorded as `missed` and stays in the book.
- *No leakage.* Models only see an information set that is cut at the issue time: prices up to
  the end of day D, weather forecasts that existed at 08:45. A test sets every later value to an
  absurd number and checks the forecast is unchanged. Weather training data uses the Open-Meteo
  Previous Runs API at the same lead time as live (48 h for D+1, 72 h for D+2), not the
  Historical Forecast API, which is close to observed weather. Grid-operator forecasts on SMARD
  are overwritten without vintages, so an hourly probe measures when they appear before they
  may be used at all.
- *Benchmarks first.* Three rules of thumb (same hour on the last known day, one week before,
  and the "similar day" rule from Lago et al. 2021, which is the reference for skill scores).
- *Models.* LEAR, a lasso-regularised autoregression per hour (implemented from the paper), and
  gradient boosting with quantile loss on weather forecasts, lagged prices and calendar. New
  models run as new versions in parallel; nothing is replaced silently.
- *Uncertainty.* Every forecast has a P10–P90 band. The site also turns the forecast into one
  user-facing answer, the cheapest 3-hour window, with probabilities that are themselves
  pre-registered and scored (hit rate against stated probability, Brier score).
- *Statistics.* MAE, RMSE, skill vs. the reference, pinball loss and band coverage, by window
  (7/30/90 days, all), horizon and situation (weekday, weekend, negative prices, spikes);
  Diebold-Mariano tests with the small-sample correction.

**Stack:** Python 3.12, uv, pandas, scikit-learn, SciPy, Parquet; GitHub Actions for forecast,
evaluation, audit, weekly release and a static bilingual site on GitHub Pages; open data from
SMARD (Bundesnetzagentur, CC BY 4.0) and Open-Meteo (CC BY 4.0). Running cost: 0 €.

## 3. Results so far

### Backtest (not pre-registered, therefore optimistic)

Rolling backtest over 363 target days (2025-10-01 to 2026-09-28), the same code with only the
data known on each day; a test checks that it reproduces the live forecast exactly.

| Model | MAE D+1 | Skill D+1 | MAE D+2 | Skill D+2 | 80 % band coverage D+1 |
|---|---|---|---|---|---|
| `lear.v1` | 22.0 | +0.33 | 29.2 | +0.23 | 0.76 |
| `naive_similar_day.v1` (reference) | 32.7 | 0 | 37.9 | 0 | 0.76 |
| `naive_last_day.v1` | 30.2 | +0.07 | 38.5 | −0.02 | 0.77 |
| `naive_weekly.v1` | 37.8 | −0.16 | 37.8 | 0.00 | 0.77 |

MAE in EUR/MWh per quarter-hour (10 EUR/MWh = 1 ct/kWh). Diebold-Mariano: LEAR better than
every rule of thumb at both horizons (p < 0.001). What the backtest also shows:

- The bands of all models are too narrow (0.76 instead of 0.80). The band is estimated from the
  previous 60–90 days and lags behind when volatility rises.
- Negative prices and spikes are where all models are worst; a model that only sees prices
  cannot know that tomorrow is sunny and windy.

### Live (from 2026-09-30)

*To be filled in from `prognosebuch headline` after at least four weeks, including missed days.*

## 4. What I would do next

- Add the grid operators' load and renewable forecasts once the probe shows at what time they
  are reliably published.
- Condition the bands on the weather situation instead of the last 60–90 days.
- Submit the forecasts to an external benchmark (Energy-Arena) for a third-party timestamp.

## 5. Transferable skills

- **Measuring, not claiming:** pre-registration, benchmark selection, proper scoring rules
  (pinball, Brier), significance testing (Diebold-Mariano), separating backtest from live.
- **Time series:** quarter-hour data with daylight-saving days, lagged features, leakage
  control, rolling-origin evaluation, quantile regression.
- **Data engineering:** reproducible pipelines with uv, typed Python (mypy strict), 80+ offline
  tests, write-once storage with checksums, CI that audits its own history.
- **Automation:** scheduled GitHub Actions with idempotent retries, failure issues that close
  themselves, weekly releases, static site deployment.
- **BI hand-off:** documented CSV/Parquet exports and a column dictionary for Power BI.
- **Communication:** a bilingual site that explains uncertainty to non-experts and states its
  own limits.
