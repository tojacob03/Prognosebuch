"""Automatic hints for why a day's forecast was bad, derived from the prices alone.

These are descriptive heuristics, not causes: the models see neither wind, sun, gas prices
nor outages, so the most frequent real cause cannot be named from this data.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

from prognosebuch.site.i18n import num, t
from prognosebuch.timeutil import TZ, day_slots_utc

LEVEL_JUMP_EUR = 25.0
BIAS_EUR = 10.0


@lru_cache(maxsize=32)
def _holidays(year: int, lang: str) -> dict[date, str]:
    language = "de" if lang == "de" else "en_US"
    return dict(holidays.country_holidays("DE", years=[year], language=language))


def holiday_name(d: date, lang: str) -> str | None:
    return _holidays(d.year, lang).get(d)


def explain_day(
    actuals: pd.Series, target: date, horizon: int, q50: pd.Series | None, lang: str
) -> list[str]:
    slots = day_slots_utc(target)
    a = actuals.reindex(slots)
    if a.isna().all():
        return []
    out: list[str] = []  # patterns in the prices themselves
    details: list[str] = []  # how the forecast missed
    last_known = target - timedelta(days=horizon)
    before = actuals.reindex(day_slots_utc(last_known))
    if before.notna().any():
        diff = a.mean() - before.mean()
        if abs(diff) >= LEVEL_JUMP_EUR:
            out.append(
                t("ex_level", lang, now=num(a.mean(), 0, lang), before=num(before.mean(), 0, lang))
            )
    neg = a[a < 0]
    if len(neg):
        out.append(t("ex_negative", lang, n=len(neg), min=num(float(neg.min()), 0, lang)))
    hist_start = day_slots_utc(target - timedelta(days=90))[0]
    idx = pd.DatetimeIndex(actuals.index)
    hist = actuals[(idx >= hist_start) & (idx < slots[0])]
    if len(hist) > 96 * 30:
        daily_max = hist.groupby(pd.DatetimeIndex(hist.index).tz_convert(TZ).date).max()
        if a.max() > np.nanquantile(daily_max.to_numpy(), 0.95):
            tmax = slots[int(np.nanargmax(a.to_numpy()))].tz_convert(TZ).strftime("%H:%M")
            out.append(t("ex_spike", lang, max=num(float(a.max()), 0, lang), time=tmax))
        g = hist.groupby(pd.DatetimeIndex(hist.index).tz_convert(TZ).date)
        usual = float(np.median((g.max() - g.min()).to_numpy()))
        spread = float(a.max() - a.min())
        if spread > 1.5 * usual:
            out.append(t("ex_spread", lang, spread=num(spread, 0, lang), usual=num(usual, 0, lang)))
    name = holiday_name(target, lang)
    if name:
        out.append(t("ex_holiday", lang, name=name))
    else:
        prev = holiday_name(target - timedelta(days=1), lang)
        if prev:
            out.append(t("ex_after_holiday", lang, name=prev))
    if len(slots) != 96:
        out.append(t("ex_dst", lang))
    if q50 is not None:
        q = q50.reindex(slots)
        err = (q - a).dropna()
        if len(err):
            bias = float(err.mean())
            if bias >= BIAS_EUR:
                details.append(t("ex_bias_high", lang, x=num(bias, 0, lang)))
            elif bias <= -BIAS_EUR:
                details.append(t("ex_bias_low", lang, x=num(-bias, 0, lang)))
            pos = int(np.argmax(np.abs(err.to_numpy())))
            when = pd.DatetimeIndex(err.index)[pos]
            details.append(
                t(
                    "ex_worst_time",
                    lang,
                    time=when.tz_convert(TZ).strftime("%H:%M"),
                    f=num(float(q.reindex(err.index).to_numpy()[pos]), 0, lang),
                    a=num(float(a.reindex(err.index).to_numpy()[pos]), 0, lang),
                )
            )
    return (out or [t("ex_none", lang)]) + details
