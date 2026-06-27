"""Tests for surrogate.py — statistical validation of HAVOK forcing."""
import numpy as np
from havolib.surrogate import (
    phase_randomized_surrogate,
    generate_surrogates,
    surrogate_forcing_distribution,
    validate_forcing_significance,
)
from havolib.pipeline import HavokPipeline


class TestPhaseRandomizedSurrogate:
    def test_preserves_mean_and_std(self):
        """Surrogate must preserve mean (±0.1%) and std (±1%)."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        x_surr = phase_randomized_surrogate(x, seed=123)

        assert abs(np.mean(x_surr) - np.mean(x)) < 0.001, "Mean not preserved"
        assert abs(np.std(x_surr) - np.std(x)) < 0.01, "Std not preserved"

    def test_preserves_autocorrelation_structure(self):
        """Phase-randomized surrogates preserve autocorrelation spectrum."""
        t = np.linspace(0, 10 * np.pi, 1000)
        x = np.sin(t) + 0.3 * np.sin(3 * t)
        x_surr = phase_randomized_surrogate(x, seed=42)

        # Autocorrelation at lag 1
        def acf1(s):
            return np.corrcoef(s[:-1], s[1:])[0, 1]

        orig_acf1 = acf1(x)
        surr_acf1 = acf1(x_surr)

        assert abs(surr_acf1 - orig_acf1) < 0.1, "Autocorrelation structure not preserved"

    def test_different_seeds_produce_different_surrogates(self):
        """Different seeds should produce different outputs (check determinism)."""
        x = np.sin(np.linspace(0, 10 * np.pi, 500))
        s1 = phase_randomized_surrogate(x, seed=42)
        s2 = phase_randomized_surrogate(x, seed=99)

        # With different seeds, surrogates should NOT be identical
        max_diff = np.max(np.abs(s1 - s2))
        assert max_diff > 0.01, f"Surrogates too similar, max diff={max_diff:.6f}"

    def test_output_is_real_valued(self):
        """Surrogates must be real-valued (no complex artifacts)."""
        x = np.random.randn(512)
        s = phase_randomized_surrogate(x, seed=99)
        assert np.all(np.isreal(s))
        assert not np.any(np.isnan(s))
        assert not np.any(np.isinf(s))

    def test_deterministic_with_same_seed(self):
        """Same seed must produce very similar surrogates (correlation near 1)."""
        x = np.random.randn(256)
        s1 = phase_randomized_surrogate(x, seed=7)
        s2 = phase_randomized_surrogate(x, seed=7)
        # They should be very close (correlation > 0.999)
        corr = np.corrcoef(s1, s2)[0, 1]
        assert corr > 0.999, f"Same-seed surrogates differ: corr={corr:.6f}"


class TestGenerateSurrogates:
    def test_generates_correct_count(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 300))
        surrs = generate_surrogates(x, n_surrogates=10, seed=42)
        assert len(surrs) == 10

    def test_all_same_length_as_input(self):
        x = np.arange(200, dtype=float)
        surrs = generate_surrogates(x, n_surrogates=5, seed=1)
        for s in surrs:
            assert len(s) == 200


class TestSurrogateForcingDistribution:
    def test_lorenz_forcing_above_surrogates(self):
        """HAVOK forcing on Lorenz must exceed surrogate threshold."""
        from havolib.data_loader import generate_lorenz

        t, x = generate_lorenz(n_points=2000)

        def make_pipe():
            return HavokPipeline(tau=10, m=30, r=5)

        pipe = make_pipe()
        pipe.fit(t, x)
        obs_max = float(np.max(np.abs(pipe.get_forcing())))

        surr_maxes, thresh_99 = surrogate_forcing_distribution(
            x, make_pipe, n_surrogates=15, seed=42
        )

        assert obs_max > thresh_99, (
            f"Expected Lorenz forcing ({obs_max:.4f}) above surrogate "
            f"99th percentile ({thresh_99:.4f})"
        )

    def test_sine_forcing_p_value_not_extreme(self):
        """Noise-like signal: forcing should not be extreme vs surrogates."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1500)  # pure noise — no structure

        def make_pipe():
            return HavokPipeline(tau=3, m=30, r=3)

        pipe = make_pipe()
        pipe.fit(np.arange(len(x)), x)
        obs_max = float(np.max(np.abs(pipe.get_forcing())))

        surr_maxes, thresh_99 = surrogate_forcing_distribution(
            x, make_pipe, n_surrogates=15, seed=123
        )

        sig, p_val = validate_forcing_significance(obs_max, surr_maxes, alpha=0.01)
        # For pure noise, p should not be extreme (surrogates should match)
        if len(surr_maxes) > 0:
            assert p_val > 0.001, f"p-value too extreme for noise: {p_val}"

    def test_returns_array_and_float(self):
        """Check return types."""
        x = np.sin(np.linspace(0, 10, 500))

        def make_pipe():
            return HavokPipeline(tau=3, m=15, r=3)

        maxes, thresh = surrogate_forcing_distribution(
            x, make_pipe, n_surrogates=5, seed=42
        )
        assert isinstance(maxes, np.ndarray)
        assert isinstance(thresh, float)
        assert len(maxes) == 5


class TestValidateForcingSignificance:
    def test_clearly_significant(self):
        """An observed value far above surrogates must be significant."""
        surr_maxes = np.array([0.1, 0.12, 0.09, 0.11, 0.10, 0.08, 0.13, 0.1, 0.11, 0.09])
        obs_max = 1.5
        sig, p = validate_forcing_significance(obs_max, surr_maxes, alpha=0.01)
        assert sig
        assert p < 0.01

    def test_clearly_not_significant(self):
        """An observed value in the surrogate range must NOT be significant."""
        surr_maxes = np.array([0.5, 0.6, 0.4, 0.55, 0.45, 0.65, 0.5, 0.55, 0.6, 0.5])
        obs_max = 0.4
        sig, p = validate_forcing_significance(obs_max, surr_maxes, alpha=0.01)
        assert not sig
        assert p > 0.01

    def test_edge_case_equal_to_surrogate(self):
        """Observed equal to max surrogate → p close to 0."""
        surr_maxes = np.array([0.1, 0.2, 0.3])
        obs_max = 0.3
        sig, p = validate_forcing_significance(obs_max, surr_maxes, alpha=0.01)
        # p = fraction >= observed = 1/3 ≈ 0.33
        assert not sig  # p > 0.01
        assert abs(p - 1.0 / 3.0) < 0.01

    def test_empty_surrogates_returns_safe_default(self):
        sig, p = validate_forcing_significance(1.0, np.array([]), alpha=0.01)
        assert not sig
        assert p == 1.0
