# Data sources

Only sources with a clear open license are used. Checked on 2026-09-29.

| Source | What | URL | License | Obligations | Used since |
|---|---|---|---|---|---|
| SMARD.de (Bundesnetzagentur) | Day-ahead price DE-LU, quarter-hourly (series 4169) | https://www.smard.de/app/chart_data/4169/DE-LU/ | CC BY 4.0 ([terms](https://www.smard.de/en/datennutzung)) | Attribution "Bundesnetzagentur \| SMARD.de"; no warranty from the provider | v0.1.0 (2026-09-30) |
| SMARD.de (Bundesnetzagentur) | Availability probe only: grid-operator forecasts of load (411), residual load (4362), generation total/wind/PV/other (122, 123, 3791, 125, 5097, 715) | same API | CC BY 4.0 | as above | probe only, not yet a model input |
| Open-Meteo | Weather forecasts (planned model input). Training: [Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) (`*_previous_day1/2`), live: Forecast API | https://open-meteo.com | Data CC BY 4.0 ([terms](https://open-meteo.com/en/terms)) | Attribution; free API only for non-commercial use; limits 600/min, 5,000/h, 10,000/day, 300,000/month | planned |

## Notes and decisions

- **SMARD keeps no vintages.** Values (in particular forecasts) are overwritten in place,
  so it is impossible to reconstruct later what was available at 09:00 on a past day. The
  hourly `probe` workflow logs how far each series reaches (branch `probe`,
  `smard_availability.csv`). A SMARD forecast series only becomes a model input once
  the log shows it is reliably available before the issue time. On 2026-09-29 at
  12:35 CEST the generation forecasts reached only the end of the current day, while the
  load forecast (411) already covered the next day.
- **Open-Meteo Historical Forecast API is not used for training.** Per its documentation
  it stitches together the first hours of each model run, which is close to observed
  weather and would leak information. The Previous Runs API provides values as they
  were forecast 1 and 2 days ahead (most models from January 2024 onwards).
- **Gas and CO2 prices** are important price drivers but no source with an open license
  was found (exchange data is proprietary). They are not scraped. This is a known
  limitation of all models here.
- **ENTSO-E Transparency Platform** requires an account and has its own terms; not used.
- **epftoolbox** (Lago et al.) is AGPL-3.0; no code is copied. The LEAR model will be
  implemented independently from the published paper and cited.
