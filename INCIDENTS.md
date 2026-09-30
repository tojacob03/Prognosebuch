# Incidents

Every operational problem that affected the book, in plain words. Missed forecasts stay in the
book as `missed`; this file explains why.

| Issue date | What happened | Effect | Fix |
|---|---|---|---|
| 2026-09-30 | GitHub started the scheduled forecast workflow at 13:03 UTC instead of 06:50/07:50 UTC (the first scheduled runs of the new repository were delayed by up to 6 hours; several were dropped). At 15:03 Berlin time the job correctly refused to issue, because the 12:00 deadline had passed. | No forecasts for 2026-10-01 (D+1) and 2026-10-02 (D+2) from this issue day; they count as missed for every model. | From 2026-09-30 a `clock` workflow waits for the timetable and dispatches forecast (08:55, 09:30, 10:30 Berlin), evaluate and probe itself; dispatches start within seconds. It hands over to a new run of itself every 5.5 hours. GitHub's cron entries remain as a backup. |
