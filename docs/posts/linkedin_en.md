<!--
DRAFT. Publish only once there are at least four weeks of live track record.
Take every value in [[…]] from `uv run prognosebuch headline` (live numbers only, never the
backtest). If LEAR is NOT better than the rule of thumb live, rewrite the second paragraph:
that is then the result.
-->

Can you forecast tomorrow's electricity price? And how do you know whether the forecast is any good?

Since 30 September I have published a forecast of German day-ahead prices every morning before the noon auction: every quarter-hour of the next two days, with an uncertainty band. Each forecast is saved as an immutable public file before the exchange publishes the prices, and scored automatically in the afternoon. Missed days count.

After [[DAYS]] days:
• My best model (LEAR, a regularised autoregression) was off by [[MAE_LEAR_D1]] EUR/MWh on average; the simple rule of thumb ("like yesterday, or like last week") by [[MAE_REF_D1]] EUR/MWh.
• That is [[SKILL_PERCENT]] % less error. The Diebold-Mariano test says: [[DM_SENTENCE]].
• Missed: [[MISSED]] forecasts.
• All models are worst on days with negative prices and price spikes; the "Where it fails" page shows which days and why.

What I learned: a backtest is easy to make look good. A forecast only becomes honest when it is fixed in advance and cannot be changed afterwards. Here a CI rule, checksums and automatic scoring take care of that.

Everything is open and runs at zero cost: Python, scikit-learn, GitHub Actions, data from SMARD (German Federal Network Agency) and Open-Meteo. For people on dynamic tariffs the site also shows the cheapest three hours, with a probability that is itself recorded in advance and checked.

Live track record: https://tojacob03.github.io/Prognosebuch/en/
Code and data: https://github.com/tojacob03/Prognosebuch

Not investment advice, no trading signals. Feedback, especially from people in energy trading and data science, is very welcome.

#DataAnalytics #EnergyTransition #Python #Forecasting #OpenData
