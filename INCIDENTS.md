# Incidents

Every operational problem that affected the book, in plain words. Missed forecasts stay in the
book as `missed`; this file explains why.

| Issue date | What happened | Effect | Fix |
|---|---|---|---|
| 2026-09-30 | GitHub started the scheduled forecast workflow at 13:03 UTC instead of 06:50/07:50 UTC (the first scheduled runs of the new repository were delayed by up to 6 hours; several were dropped). At 15:03 Berlin time the job correctly refused to issue, because the 12:00 deadline had passed. | No forecasts for 2026-10-01 (D+1) and 2026-10-02 (D+2) from this issue day; they count as missed for every model. | From 2026-09-30 a `clock` workflow waits for the timetable and dispatches forecast (08:55, 09:30, 10:30 Berlin), evaluate and probe itself; dispatches start within seconds. It hands over to a new run of itself every 5.5 hours. GitHub's cron entries remain as a backup. |
| 2026-10-10 | SMARD's API had no prices for delivery day 2026-10-10 (normally published on 2026-10-09 around 12:45; still missing on 2026-10-10 at 16:00 UTC, while the days before and after were complete). The forecast job ran on time at 08:55, 09:30 and 10:30, but `naive_last_day.v1`, `lear.v1` and `gbm.v1` need the issue day's prices and refused to forecast with a missing day. | `naive_weekly.v1` and `naive_similar_day.v1` were issued; the other three models missed issue day 2026-10-10 (targets 2026-10-11 and 2026-10-12). | Quarter-hours missing on SMARD are now filled from the Energy-Charts API (Fraunhofer ISE), which republishes the same SMARD series; values on all shared quarter-hours were identical. Manifests record any filled values; the evaluate job logs them in `actuals/day_ahead_price_de_lu/fallback_log.csv`. |
