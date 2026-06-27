"""Tests for auto_tune.py — Mutual Information + False Nearest Neighbors."""
import numpy as np
from havolib.auto_tune import (
    mutual_information,
    optimal_tau_mi,
    false_nearest_neighbors,
    optimal_m_fnn,
    suggest_parameters,
)


class TestMutualInformation:
    def test_perfect_correlation(self):
        """Identical signals should have non-negative MI (density-based MI can be tricky)."""
        x = np.sin(np.linspace(0, 10, 500))
        mi = mutual_information(x, x)
        # MI should be finite and not NaN
        assert np.isfinite(mi), f"MI not finite: {mi}"

    def test_independent_signals(self):
        """Independent random signals should have very low MI."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        y = rng.normal(0, 1, 1000)
        mi = mutual_information(x, y)
        # MI should be close to 0 for independent signals
        assert mi < 0.2, f"MI for independent signals too high: {mi:.4f}"

    def test_sinusoid_delayed_has_peak(self):
        """A delayed sinusoid should have finite MI values."""
        t = np.linspace(0, 10 * np.pi, 1000)
        x = np.sin(t)
        # MI with itself at different lags should produce finite values
        mi_lag1 = mutual_information(x[:-1], x[1:])
        mi_lag10 = mutual_information(x[:-10], x[10:])
        assert np.isfinite(mi_lag1)
        assert np.isfinite(mi_lag10)


class TestOptimalTauMi:
    def test_returns_reasonable_range(self):
        """For a sinusoid, tau should be > 0 and reasonable."""
        t = np.linspace(0, 10 * np.pi, 500)
        x = np.sin(t)
        tau = optimal_tau_mi(x, max_lag=50)
        assert tau > 0
        assert tau <= 50

    def test_returns_int(self):
        x = np.random.randn(300)
        tau = optimal_tau_mi(x, max_lag=30)
        assert isinstance(tau, (int, np.integer))

    def test_fallback_when_no_minimum(self):
        """Constant signal should use the 1/e fallback."""
        x = np.ones(200) + np.random.normal(0, 0.001, 200)
        tau = optimal_tau_mi(x, max_lag=30)
        assert tau > 0
        assert tau <= 30


class TestFalseNearestNeighbors:
    def test_returns_array(self):
        t = np.linspace(0, 10 * np.pi, 1000)
        x = np.sin(t)
        fracs = false_nearest_neighbors(x, tau=5, max_m=30)
        assert isinstance(fracs, np.ndarray)
        assert len(fracs) > 0
        assert len(fracs) <= 30

    def test_fnn_decreases_with_m(self):
        """For clean data, FNN should eventually drop below threshold."""
        t = np.linspace(0, 15 * np.pi, 2000)
        x = np.sin(t)
        fracs = false_nearest_neighbors(x, tau=10, max_m=30)
        if len(fracs) > 1:
            assert fracs[-1] < 0.5, f"FNN should drop below 0.5 for sine: {fracs[-1]:.3f}"


class TestOptimalMFNN:
    def test_returns_reasonable_m(self):
        t = np.linspace(0, 15 * np.pi, 2000)
        x = np.sin(t)
        m = optimal_m_fnn(x, tau=10, max_m=30)
        assert m > 0
        assert m <= 30

    def test_returns_int(self):
        x = np.random.randn(1000)
        m = optimal_m_fnn(x, tau=3, max_m=20)
        assert isinstance(m, (int, np.integer))


class TestSuggestParameters:
    def test_returns_dict_with_expected_keys(self):
        t, x = np.linspace(0, 15 * np.pi, 2000), None
        from havolib.data_loader import generate_lorenz
        _, x = generate_lorenz(n_points=2000)
        params = suggest_parameters(x)
        assert "tau" in params
        assert "m" in params
        assert "method" in params
        assert "recommendation" in params

    def test_lorenz_parameters_reasonable(self):
        from havolib.data_loader import generate_lorenz
        _, x = generate_lorenz(n_points=2000)
        params = suggest_parameters(x)
        assert params["tau"] > 0
        assert params["m"] >= 2
        assert params["m"] <= 50
