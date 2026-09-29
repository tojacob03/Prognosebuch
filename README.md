# probe

Hourly log of how far each SMARD series reaches (written by `.github/workflows/probe.yml` on `main`).
Used to measure when grid-operator forecasts for D+1 become available. Data: Bundesnetzagentur | SMARD.de, CC BY 4.0.

Columns: probed_at_utc, series (smard:<filter>:<region>:<resolution>), description, last_value_start_utc, last_value_start_local.
