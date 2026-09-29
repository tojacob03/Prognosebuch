"""Server-side SVG charts. Colours come from CSS classes, so light and dark themes work."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from html import escape

import numpy as np
import pandas as pd

from prognosebuch.site.i18n import num, short_date
from prognosebuch.timeutil import TZ


@dataclass(frozen=True)
class Frame:
    width: int
    height: int
    left: int = 46
    right: int = 10
    top: int = 16
    bottom: int = 30

    @property
    def pw(self) -> float:
        return self.width - self.left - self.right

    @property
    def ph(self) -> float:
        return self.height - self.top - self.bottom


WIDE = Frame(960, 380)
NARROW = Frame(440, 320, left=40)
SMALL = Frame(440, 220, left=40, top=10)


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + 1e-9:
        ticks.append(round(v, 6))
        v += step
    return ticks


def _y_scale(values: list[float], f: Frame, n_ticks: int) -> tuple[list[float], float, float]:
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.06 or 5
    ticks = nice_ticks(lo - pad, hi + pad, n_ticks)
    return ticks, ticks[0], ticks[-1]


def day_chart(
    df: pd.DataFrame,
    lang: str,
    frame: Frame = WIDE,
    actual: pd.Series | None = None,
    highlight: tuple[int, int] | None = None,
    highlight_label: str | None = None,
    title: str = "",
    chart_id: str = "",
) -> str:
    """Quarter-hour ladder: one bar P10-P90 per quarter-hour, median tick, actual as a line.

    ``df`` has columns delivery_start_utc, q10, q50, q90 in time order.
    """
    f = frame
    n = len(df)
    q10 = df["q10"].to_numpy(dtype=float)
    q50 = df["q50"].to_numpy(dtype=float)
    q90 = df["q90"].to_numpy(dtype=float)
    act = None if actual is None else actual.to_numpy(dtype=float)
    vals = [*q10, *q90] + ([] if act is None else [v for v in act if not np.isnan(v)])
    ticks, y0, y1 = _y_scale(vals, f, 5 if f is WIDE else 4)

    def y(v: float) -> float:
        return f.top + f.ph * (1 - (v - y0) / (y1 - y0))

    bw = f.pw / n

    def x(i: float) -> float:
        return f.left + i * bw

    parts = [
        f'<svg class="chart day-chart" viewBox="0 0 {f.width} {f.height}" role="img" '
        f'aria-labelledby="{chart_id}-t" data-chart="{chart_id}">',
        f'<title id="{chart_id}-t">{escape(title)}</title>',
    ]
    if highlight is not None:
        j0, w = highlight
        parts.append(
            f'<rect class="c-mark" x="{x(j0):.1f}" y="{f.top}" width="{w * bw:.1f}" '
            f'height="{f.ph:.1f}"/>'
        )
        if highlight_label:
            anchor = "start" if j0 < n * 0.7 else "end"
            tx = x(j0) + 4 if anchor == "start" else x(j0 + w) - 4
            parts.append(
                f'<text class="c-mark-label" x="{tx:.1f}" y="{f.top + 14}" '
                f'text-anchor="{anchor}">{escape(highlight_label)}</text>'
            )
    for tv in ticks:
        cls = "c-zero" if abs(tv) < 1e-9 else "c-grid"
        parts.append(
            f'<line class="{cls}" x1="{f.left}" x2="{f.width - f.right}" y1="{y(tv):.1f}" '
            f'y2="{y(tv):.1f}"/>'
        )
        parts.append(
            f'<text class="c-tick" x="{f.left - 6}" y="{y(tv) + 4:.1f}" text-anchor="end">'
            f"{num(tv, 0, lang)}</text>"
        )
    local = pd.DatetimeIndex(df["delivery_start_utc"]).tz_convert(TZ)
    step = 3 if f is WIDE else 6
    for i, ts in enumerate(local):
        if ts.minute == 0 and ts.hour % step == 0:
            parts.append(
                f'<line class="c-grid c-vgrid" x1="{x(i):.1f}" x2="{x(i):.1f}" y1="{f.top}" '
                f'y2="{f.top + f.ph:.1f}"/>'
            )
            parts.append(
                f'<text class="c-tick" x="{x(i):.1f}" y="{f.height - 10}" text-anchor="middle">'
                f"{ts:%H}:00</text>"
            )
    inset = bw * 0.14
    for i in range(n):
        parts.append(
            f'<rect class="c-band" x="{x(i) + inset:.2f}" y="{y(q90[i]):.2f}" '
            f'width="{bw - 2 * inset:.2f}" height="{max(y(q10[i]) - y(q90[i]), 0.8):.2f}"/>'
        )
    med = " ".join(f"M{x(i) + inset:.2f},{y(q50[i]):.2f}h{bw - 2 * inset:.2f}" for i in range(n))
    parts.append(f'<path class="c-med" d="{med}"/>')
    if act is not None and not np.isnan(act).all():
        pts = []
        for i in range(n):
            if not np.isnan(act[i]):
                pts.append(f"{x(i):.2f},{y(act[i]):.2f} {x(i + 1):.2f},{y(act[i]):.2f}")
        parts.append(f'<polyline class="c-act" points="{" ".join(pts)}"/>')
    for i in range(n):
        parts.append(
            f'<rect class="c-hit" data-i="{i}" x="{x(i):.2f}" y="{f.top}" width="{bw:.2f}" '
            f'height="{f.ph:.1f}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def line_chart(
    series: dict[str, pd.Series],
    lang: str,
    frame: Frame = WIDE,
    emphasis: str | None = None,
    reference: str | None = None,
    title: str = "",
    chart_id: str = "",
) -> str:
    """Daily values per model; the last point of each line is labelled directly."""
    f = Frame(frame.width, frame.height, left=frame.left, right=150, top=frame.top)
    all_dates = sorted({d for s in series.values() for d in s.index})
    if not all_dates:
        return ""
    vals = [float(v) for s in series.values() for v in s.to_numpy() if not np.isnan(v)]
    ticks, _, y1 = _y_scale([0.0, *vals], f, 4)
    y0 = 0.0
    ticks = [tv for tv in ticks if tv >= 0]
    d0, d1 = all_dates[0], all_dates[-1]
    span = max((d1 - d0).days, 1)

    def x(d: date) -> float:
        return f.left + f.pw * (d - d0).days / span

    def y(v: float) -> float:
        return f.top + f.ph * (1 - (v - y0) / (y1 - y0))

    parts = [
        f'<svg class="chart line-chart" viewBox="0 0 {f.width} {f.height}" role="img" '
        f'aria-labelledby="{chart_id}-t">',
        f'<title id="{chart_id}-t">{escape(title)}</title>',
    ]
    for tv in ticks:
        parts.append(
            f'<line class="c-grid" x1="{f.left}" x2="{f.left + f.pw:.1f}" y1="{y(tv):.1f}" '
            f'y2="{y(tv):.1f}"/>'
            f'<text class="c-tick" x="{f.left - 6}" y="{y(tv) + 4:.1f}" text-anchor="end">'
            f"{num(tv, 0, lang)}</text>"
        )
    n_labels = min(6, len(all_dates))
    for d in [
        all_dates[round(k * (len(all_dates) - 1) / max(n_labels - 1, 1))] for k in range(n_labels)
    ]:
        parts.append(
            f'<text class="c-tick" x="{x(d):.1f}" y="{f.height - 10}" text-anchor="middle">'
            f"{escape(short_date(d, lang))}</text>"
        )
    labels = []
    for key, s in series.items():
        s = s.dropna().sort_index()
        if s.empty:
            continue
        cls = "c-line"
        if key == emphasis:
            cls += " c-line-em"
        if key == reference:
            cls += " c-line-ref"
        pts = " ".join(
            f"{x(d):.1f},{y(float(v)):.1f}"
            for d, v in zip(list(s.index), s.to_numpy(), strict=True)
        )
        parts.append(f'<polyline class="{cls}" points="{pts}"/>')
        if len(s) == 1:
            d, v = s.index[0], float(s.iloc[0])
            parts.append(f'<circle class="{cls}" cx="{x(d):.1f}" cy="{y(v):.1f}" r="3"/>')
        labels.append((y(float(s.iloc[-1])), key, cls))
    labels.sort()
    last_y = -1e9
    for ly, key, cls in labels:
        ly = max(ly, last_y + 14)
        last_y = ly
        parts.append(
            f'<text class="c-line-label {cls}" x="{f.left + f.pw + 8:.1f}" y="{ly + 4:.1f}">'
            f"{escape(key)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)
