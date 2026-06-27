"""Tests for edge_of_chaos.py — Lyapunov, correlation dimension, critical slowing down."""
import numpy as np
from havolib.edge_of_chaos import (
    largest_lyapunov_exponent,
    critical_slowing_down,
    edge_of_chaos_score,
)


class TestLargestLyapunovExponent:
    def test_lorenz_positive_lle(self):
        """Lorenz attractor must have positive Lyapunov exponent (chaos)."""
        from havolib.data_loader import generate_lorenz
        _, x = generate_lorenz(n_points=3000)
        lle = largest_lyapunov_exponent(x, tau=10, m=5, dt=0.01)
        assert lle > 0.001, f"Expected positive LLE for Lorenz, got {lle:.6f}"

    def test_sine_near_zero_lle(self):
        """Sine should have near-zero or negative LLE (periodic)."""
        t = np.linspace(0, 20 * np.pi, 2000)
        x = np.sin(t)
        lle = largest_lyapunov_exponent(x, tau=5, m=3, dt=0.01)
        # Periodic → LLE should be close to zero
        assert lle < 0.1, f"Expected near-zero LLE for sine, got {lle:.6f}"

    def test_short_signal_returns_zero(self):
        x = np.random.randn(30)
        lle = largest_lyapunov_exponent(x, tau=1, m=3)
        assert lle == 0.0


class TestCriticalSlowingDown:
    def test_returns_value_between_0_and_1(self):
        x = np.random.randn(500)
        csd = critical_slowing_down(x, lag=1)
        assert 0.0 <= csd <= 1.0

    def test_highly_autocorrelated_signal(self):
        """Slowly varying signal should have high CSD."""
        t = np.linspace(0, 10 * np.pi, 500)
        x = np.sin(t) + 0.1 * np.random.randn(500)
        csd = critical_slowing_down(x, lag=1)
        assert csd > 0.5, f"Expected high CSD for sinusoid, got {csd:.3f}"

    def test_white_noise_low_csd(self):
        x = np.random.randn(1000)
        csd = critical_slowing_down(x, lag=1)
        assert csd < 0.3, f"Expected low CSD for noise, got {csd:.3f}"


class TestEdgeOfChaosScore:
    def test_returns_expected_keys(self):
        from havolib.data_loader import generate_lorenz
        _, x = generate_lorenz(n_points=2000)
        eoc = edge_of_chaos_score(x)
        assert "largest_lyapunov_exponent" in eoc
        assert "critical_slowing_down_lag1" in eoc
        assert "edge_of_chaos_score" in eoc
        assert "interpretation" in eoc
        assert 0.0 <= eoc["edge_of_chaos_score"] <= 1.0

    def test_lorenz_changes_interpretation(self):
        from havolib.data_loader import generate_lorenz
        _, x = generate_lorenz(n_points=2000)
        eoc = edge_of_chaos_score(x)
        # Lorenz should be either chaotic or edge-of-chaos
        assert eoc["interpretation"] in [
            "🔥 EDGE OF CHAOS — rich adaptive dynamics",
            "🌪️ CHAOTIC — random, unstructured",
            "⚡ TRANSITION — approaching the edge",
            "🧊 ORDERED — rigid, predictable",
        ]
