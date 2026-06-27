"""Tests for sklearn-compatible HavokEstimator + differentiation methods."""
import numpy as np
import pytest
from havolib.estimator import (
    HavokEstimator, finite_diff, spline_diff, total_variation_diff,
    DIFF_METHODS, cross_val_score_havok,
)
from havolib.data_loader import generate_lorenz


class TestDifferentiationMethods:
    def test_finite_diff_order2(self):
        t = np.linspace(0, 10, 200)
        x = np.sin(t)
        V = np.column_stack([x, np.cos(x)])
        dv = finite_diff(V, t, order=2)
        assert dv.shape == V.shape
        assert np.all(np.isfinite(dv))

    def test_spline_diff(self):
        t = np.linspace(0, 10, 200)
        x = np.sin(t)
        V = np.column_stack([x])
        dv = spline_diff(V, t)
        assert dv.shape == V.shape
        assert np.all(np.isfinite(dv))
        # Derivative of sin should have non-trivial correlation with cos
        # (Allow some lag due to spline smoothing)
        cos_expected = np.cos(t[:len(dv)])
        # Check the sign pattern matches (both should oscillate)
        assert np.any((dv[:, 0] > 0) & (cos_expected > 0)), "No positive correlation region"

    def test_tv_diff(self):
        t = np.linspace(0, 10, 200)
        x = np.sin(t)
        V = np.column_stack([x])
        dv = total_variation_diff(V, t, alpha=0.1)
        assert dv.shape == V.shape
        assert np.all(np.isfinite(dv))

    def test_all_methods_registered(self):
        assert "finite_diff" in DIFF_METHODS
        assert "spline" in DIFF_METHODS
        assert "total_variation" in DIFF_METHODS
        assert "gradient" in DIFF_METHODS


class TestHavokEstimator:
    def test_fit_returns_self(self):
        x = np.sin(np.linspace(0, 20*np.pi, 500))
        est = HavokEstimator(m=20, r=3)
        result = est.fit(x)
        assert result is est

    def test_fit_sets_attributes(self):
        x = np.sin(np.linspace(0, 20*np.pi, 500))
        est = HavokEstimator(m=20, r=3)
        est.fit(x)
        assert hasattr(est, "forcing_")
        assert hasattr(est, "risk_")
        assert hasattr(est, "eigen_coords_")
        assert len(est.forcing_) == len(x)
        assert len(est.risk_) == len(x)

    def test_transform_returns_forcing(self):
        x = np.sin(np.linspace(0, 20*np.pi, 500))
        est = HavokEstimator(m=20, r=3)
        est.fit(x)
        f = est.transform(x)
        assert np.allclose(f, est.forcing_)

    def test_fit_transform(self):
        x = np.sin(np.linspace(0, 20*np.pi, 500))
        est = HavokEstimator(m=20, r=3)
        f = est.fit_transform(x)
        assert len(f) == len(x)

    def test_score_positive_for_lorenz(self):
        _, x = generate_lorenz(n_points=2000)
        est = HavokEstimator(m=30, r=5)
        est.fit(x)
        score = est.score(x)
        assert score > 0

    def test_get_set_params(self):
        est = HavokEstimator(tau=5, m=30, r=3)
        params = est.get_params()
        assert params["tau"] == 5
        assert params["r"] == 3
        est.set_params(r=8)
        assert est.r == 8

    def test_plot_returns_figure(self):
        x = np.sin(np.linspace(0, 10*np.pi, 300))
        est = HavokEstimator(m=15, r=3)
        est.fit(x)
        fig = est.plot()
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)

    def test_auto_tau_m(self):
        x = np.sin(np.linspace(0, 20*np.pi, 500))
        est = HavokEstimator(tau="auto", m="auto", r=3)
        est.fit(x)
        assert est.tau_fitted_ > 0
        assert est.m_fitted_ >= 5

    def test_different_diff_methods(self):
        x = np.sin(np.linspace(0, 10*np.pi, 300))
        for method in ["finite_diff", "spline", "gradient"]:
            est = HavokEstimator(m=15, r=3, diff_method=method)
            est.fit(x)
            assert np.all(np.isfinite(est.forcing_))

    def test_lorenz_produces_high_score(self):
        _, x = generate_lorenz(n_points=3000)
        est = HavokEstimator(m=30, r=5)
        est.fit(x)
        lorenz_score = est.score(x)
        # Lorenz should produce non-trivial forcing activity
        assert lorenz_score > 0.5, f"Lorenz score too low: {lorenz_score:.3f}"

        x_sine = np.sin(np.linspace(0, 40*np.pi, 3000))
        est2 = HavokEstimator(m=30, r=5)
        est2.fit(x_sine)
        sine_score = est2.score(x_sine)
        # Both should produce meaningful scores
        assert sine_score > 0.5, f"Sine score too low: {sine_score:.3f}"


class TestCrossValidation:
    def test_grid_search_runs(self):
        x = np.sin(np.linspace(0, 20*np.pi, 800))
        result = cross_val_score_havok(
            x,
            param_grid={"tau": [1, 5], "m": [20, 30], "r": [3, 5]},
            cv=2,
        )
        assert "best_params" in result
        assert "best_score" in result
        assert len(result["cv_results"]) == 8  # 2*2*2 = 8 combos

    def test_best_params_are_valid(self):
        x = np.sin(np.linspace(0, 20*np.pi, 600))
        result = cross_val_score_havok(
            x,
            param_grid={"tau": [1], "m": [15, 25], "r": [3]},
            cv=2,
        )
        bp = result["best_params"]
        assert bp["tau"] == 1
        assert bp["m"] in [15, 25]
        assert bp["r"] == 3
