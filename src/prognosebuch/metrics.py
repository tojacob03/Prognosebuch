"""Forecast accuracy metrics and the Diebold-Mariano test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]


def pinball(y: FloatArray, q: FloatArray, tau: float) -> FloatArray:
    """Pinball (quantile) loss of quantile forecast ``q`` at level ``tau``."""
    diff = y - q
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def mean_pinball(y: FloatArray, q10: FloatArray, q50: FloatArray, q90: FloatArray) -> FloatArray:
    """Average pinball loss over the 10 %, 50 % and 90 % quantiles, per observation."""
    return (pinball(y, q10, 0.1) + pinball(y, q50, 0.5) + pinball(y, q90, 0.9)) / 3.0


def mae(err: FloatArray) -> float:
    return float(np.mean(np.abs(err)))


def rmse(err: FloatArray) -> float:
    return float(np.sqrt(np.mean(np.square(err))))


def skill(metric_model: float, metric_reference: float) -> float:
    """1 - model/reference: > 0 means better than the reference, 0 equal, < 0 worse."""
    if metric_reference == 0:
        return float("nan")
    return 1.0 - metric_model / metric_reference


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float
    n: int
    mean_loss_diff: float


def diebold_mariano(loss_a: FloatArray, loss_b: FloatArray, h: int = 1) -> DMResult:
    """Two-sided Diebold-Mariano test with the Harvey-Leybourne-Newbold correction.

    ``loss_a`` and ``loss_b`` are paired loss series (e.g. daily MAE). A negative
    statistic means A has lower loss than B. ``h`` is the forecast horizon in steps of
    the series; autocovariances up to lag h-1 enter the variance.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = d.size
    if n < 3:
        return DMResult(float("nan"), float("nan"), n, float(np.mean(d)) if n else float("nan"))
    dbar = float(np.mean(d))
    dc = d - dbar
    gamma = [float(np.dot(dc[k:], dc[: n - k]) / n) for k in range(h)]
    var = (gamma[0] + 2.0 * sum(gamma[1:])) / n
    if var <= 0:
        return DMResult(float("nan"), float("nan"), n, dbar)
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    p = float(2.0 * stats.t.sf(abs(stat), df=n - 1))
    return DMResult(stat, p, n, dbar)
