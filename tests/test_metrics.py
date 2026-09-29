import numpy as np
import pytest

from prognosebuch.metrics import diebold_mariano, mae, mean_pinball, pinball, rmse, skill


def test_pinball_known_values() -> None:
    y = np.array([10.0, 10.0])
    assert pinball(y, np.array([8.0, 12.0]), 0.9).tolist() == pytest.approx([1.8, 0.2])
    assert pinball(y, np.array([8.0, 12.0]), 0.5).tolist() == pytest.approx([1.0, 1.0])


def test_mean_pinball_of_exact_forecast_is_zero() -> None:
    y = np.array([-50.0, 0.0, 300.0])
    assert mean_pinball(y, y, y, y).tolist() == [0.0, 0.0, 0.0]


def test_mae_rmse_skill() -> None:
    e = np.array([3.0, -4.0])
    assert mae(e) == 3.5
    assert rmse(e) == pytest.approx(np.sqrt(12.5))
    assert skill(5.0, 10.0) == 0.5
    assert np.isnan(skill(1.0, 0.0))


def test_diebold_mariano_detects_clearly_better_model() -> None:
    rng = np.random.default_rng(1)
    good = np.abs(rng.normal(0, 5, 60))
    bad = good + 3 + np.abs(rng.normal(0, 1, 60))
    r = diebold_mariano(good, bad, h=1)
    assert r.statistic < 0 and r.p_value < 0.01 and r.n == 60
    r2 = diebold_mariano(bad, good, h=2)
    assert r2.statistic > 0


def test_diebold_mariano_equal_losses_is_undefined() -> None:
    x = np.ones(20)
    assert np.isnan(diebold_mariano(x, x).p_value)
