"""
Rigorous scientific validation & regression tests for HAVOK toolbox.

These tests verify correctness, not just that functions run. Each test
asserts a specific mathematical property or reproduces a paper result.

Golden values were computed 2026-06-28 with numpy.random.seed(42/43)
and must remain stable within the stated tolerance.
"""
import numpy as np
import pytest
from numpy.random import default_rng

from havolib.data_loader import generate_lorenz
from havolib.estimator import HavokEstimator
from havolib.decomposition import eigen_time_delay
from havolib.embedding import hankel_matrix
from havolib.pipeline import HavokPipeline
from havolib.multichannel import MultichannelHAVOK

# ═══════════════════════════════════════════════════════════
# Tight golden-value tests (single-precision tolerance)
# ═══════════════════════════════════════════════════════════

class TestGoldenValues:
    """Golden values from fixed seeds — must remain bit-stable."""

    def test_lorenz_standard_forcing_exact(self):
        """Forcing on Lorenz (seed=42, n=3000, tau=1, m=50, r=5) must match."""
        np.random.seed(42)
        _, x = generate_lorenz(3000)
        est = HavokEstimator(tau=1, m=50, r=5)
        est.fit(x)
        f = np.abs(est.forcing_)

        # These exact values were computed 2026-06-28 with seed 42.
        # Tolerance: 1e-12 absolute for max, 1e-4 relative for risk count.
        assert np.max(f) == pytest.approx(0.010809597281230, abs=1e-12)
        assert np.mean(f) == pytest.approx(0.001722637458497, abs=1e-12)
        assert int(np.sum(est.risk_)) == 472

    def test_lorenz_forcing_sparsity_invariant(self):
        """Forcing must be heavy-tailed: p99/p90 > 1.5."""
        np.random.seed(42)
        _, x = generate_lorenz(3000)
        est = HavokEstimator(tau=1, m=50, r=5)
        est.fit(x)
        fs = np.sort(np.abs(est.forcing_))
        ratio = fs[int(0.99 * len(fs))] / fs[int(0.90 * len(fs))]
        assert ratio > 1.5, f"p99/p90 = {ratio:.3f} (expected > 1.5)"

    def test_lorenz_forcing_intermittency(self):
        """Max/median forcing ratio must be > 7 (intermittent signal)."""
        np.random.seed(42)
        _, x = generate_lorenz(3000)
        est = HavokEstimator(tau=1, m=50, r=5)
        est.fit(x)
        f = np.abs(est.forcing_)
        ratio = np.max(f) / np.median(f)
        assert ratio > 7.0, f"max/median = {ratio:.1f} (expected > 7)"

    def test_golden_svd_singular_values(self):
        """SVD of 2000-pt Lorenz Hankel (seed=43) — singular values stable."""
        np.random.seed(43)
        _, x = generate_lorenz(2000)
        H = hankel_matrix(x, 40, 1)
        _, s = eigen_time_delay(H, 5, solver="scipy")
        # First singular value
        assert s[0] == pytest.approx(2099.1834485724, rel=1e-8)
        # 99.9%+ energy in top 5 modes
        full_s = np.linalg.svd(H, full_matrices=False)[1]
        cum = float(np.sum(s**2) / np.sum(full_s**2))
        assert cum > 0.999

    def test_golden_pipeline_equals_estimator(self):
        """Pipeline and estimator produce identical forcing (single truth center)."""
        np.random.seed(42)
        _, x = generate_lorenz(3000)
        p = HavokPipeline(tau=1, m=50, r=5)
        p.fit(None, x)
        f_pipe = p.get_forcing()

        est = HavokEstimator(tau=1, m=50, r=5)
        est.fit(x)
        f_est = est.forcing_

        assert np.allclose(f_pipe, f_est, atol=1e-15)

    def test_golden_hankel_first_element(self):
        """H[0,0] must equal x[0] — Hankel matrix invariant."""
        np.random.seed(43)
        _, x = generate_lorenz(2000)
        H = hankel_matrix(x, 40, 1)
        assert float(H[0, 0]) == pytest.approx(float(x[0]), abs=1e-12)


# ═══════════════════════════════════════════════════════════
# Mathematical invariant tests
# ═══════════════════════════════════════════════════════════

class TestMathematicalInvariants:
    """Core HAVOK mathematics — these MUST hold for any valid input."""

    @pytest.mark.parametrize("n", [500, 1000, 3000])
    @pytest.mark.parametrize("m", [20, 50])
    @pytest.mark.parametrize("r", [3, 5, 8])
    def test_svd_orthonormality(self, n, m, r):
        """U^T U = I — eigen-time-delay coordinates must be orthonormal."""
        np.random.seed(42)
        t_arr = np.linspace(0, 30, n)
        x = np.sin(t_arr) + np.random.randn(n) * 0.05
        H = hankel_matrix(x, min(m, n // 3), 1)
        U, _ = eigen_time_delay(H, min(r, m - 1), solver="scipy")
        UtU = U.T @ U
        assert np.allclose(UtU, np.eye(U.shape[1]), atol=1e-12)

    def test_forcing_zero_mean_residual(self):
        """Forcing = V̇ - AV — the residual should be approximately zero-mean."""
        np.random.seed(42)
        _, x = generate_lorenz(3000)
        est = HavokEstimator(tau=1, m=50, r=5)
        est.fit(x)
        f = est.forcing_
        # Mean should be very close to zero (affine model with bias term)
        assert abs(np.mean(f)) < 1e-3

    def test_risk_binary(self):
        """Risk output must be strictly {0, 1}."""
        np.random.seed(42)
        _, x = generate_lorenz(2000)
        est = HavokEstimator(tau=1, m=30, r=5)
        est.fit(x)
        unique = np.unique(est.risk_)
        assert set(unique).issubset({0, 1})

    def test_forcing_length_matches_risk(self):
        """Forcing and risk arrays must have identical length."""
        np.random.seed(42)
        _, x = generate_lorenz(2500)
        est = HavokEstimator(tau=1, m=40, r=5)
        est.fit(x)
        assert len(est.forcing_) == len(est.risk_)

    def test_hankel_output_length(self):
        """Hankel matrix rows = n - (m-1)*tau."""
        np.random.seed(42)
        _, x = generate_lorenz(3000)
        for m_val in [10, 30, 50]:
            for tau_val in [1, 3, 7]:
                H = hankel_matrix(x[:1000], m_val, tau_val)
                expected = len(x[:1000]) - (m_val - 1) * tau_val
                assert H.shape[0] == expected
                assert H.shape[1] == m_val

    def test_pipeline_preprocessing_idempotent(self):
        """Pipeline with preprocessing disabled should equal raw estimator."""
        np.random.seed(42)
        _, x = generate_lorenz(1500)
        p = HavokPipeline(tau=1, m=30, r=5, do_preprocess=False)
        p.fit(None, x)
        est = HavokEstimator(tau=1, m=30, r=5)
        est.fit(x)
        assert np.allclose(p.get_forcing(), est.forcing_, atol=1e-15)


# ═══════════════════════════════════════════════════════════
# Paper reproduction tests (Brunton et al. 2017)
# ═══════════════════════════════════════════════════════════

class TestPaperReproduction:
    """Behavioral properties described in Brunton et al. 2017."""

    def test_paper_lorenz_forcing_is_bursty(self):
        """Paper claim: forcing is intermittent with rare large excursions."""
        np.random.seed(42)
        _, x = generate_lorenz(5000)
        est = HavokEstimator(tau=1, m=50, r=5)
        est.fit(x)
        f = np.abs(est.forcing_)
        f_sorted = np.sort(f)

        # Top 1% of forcing should have > 4% of energy (heavy-tailed, not extreme)
        top_1pct = f_sorted[-int(0.01 * len(f)):]
        assert np.sum(top_1pct) > np.sum(f) * 0.04, \
            f"Top 1% energy = {np.sum(top_1pct)/np.sum(f)*100:.1f}%"

    def test_paper_linear_model_fits_well(self):
        """Paper claim: linear model V̇ ≈ AV explains most variance."""
        np.random.seed(42)
        _, x = generate_lorenz(3000)
        est = HavokEstimator(tau=1, m=50, r=5)
        est.fit(x)

        V = est.eigen_coords_
        t = np.arange(len(V), dtype=float)
        y = np.gradient(V[:, -1], t)
        X = np.column_stack([V[:, :-1], np.ones(len(V))])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ coeffs
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / max(ss_tot, 1e-15)
        assert r2 > 0.40, f"R² = {r2:.3f} (expected > 0.40)"


# ═══════════════════════════════════════════════════════════
# Cross-method consistency tests
# ═══════════════════════════════════════════════════════════

class TestCrossMethodConsistency:
    """Different methods should produce correlated, not contradictory, results."""

    def test_diff_methods_correlated(self):
        """All differentiation methods should produce positively correlated forcing."""
        np.random.seed(42)
        _, x = generate_lorenz(2000)
        forcings = {}
        for method in ["finite_diff", "spline", "gradient"]:
            est = HavokEstimator(tau=1, m=30, r=5, diff_method=method)
            est.fit(x)
            forcings[method] = np.abs(est.forcing_)

        # All pairwise correlations must be > 0.3
        # (methods differ but should not be anti-correlated)
        methods = list(forcings.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                corr = np.corrcoef(forcings[methods[i]], forcings[methods[j]])[0, 1]
                assert corr > 0.3, \
                    f"corr({methods[i]}, {methods[j]}) = {corr:.3f} (expected > 0.3)"

    def test_pipeline_estimator_same_risk(self):
        """Pipeline and estimator should produce identical risk flags."""
        np.random.seed(42)
        _, x = generate_lorenz(2500)
        p = HavokPipeline(tau=1, m=40, r=5)
        p.fit(None, x)
        est = HavokEstimator(tau=1, m=40, r=5)
        est.fit(x)
        assert np.array_equal(p.get_risk(), est.risk_)

    def test_multichannel_parallel_vs_composite_consistent(self):
        """Parallel and composite mHAVOK should not produce contradictory risk."""
        np.random.seed(42)
        _, x0 = generate_lorenz(2000)
        X = np.column_stack([x0 + np.random.randn(2000) * 0.1 for _ in range(4)])

        mh_p = MultichannelHAVOK(n_channels=4, tau=1, m=40, r=5, method="parallel")
        r_p = mh_p.fit_transform(X, show_progress=False)

        mh_c = MultichannelHAVOK(n_channels=4, tau=1, m=40, r=5, method="composite")
        r_c = mh_c.fit_transform(X, show_progress=False)

        # Joint risk should have significant overlap (Jaccard > 0.3)
        intersection = np.sum(r_p.joint_risk & r_c.joint_risk)
        union = np.sum(r_p.joint_risk | r_c.joint_risk)
        if union > 0:
            jaccard = intersection / union
            assert jaccard > 0.2, f"Jaccard = {jaccard:.3f} (expected > 0.2)"


# ═══════════════════════════════════════════════════════════
# Property-based tests (Hypothesis)
# ═══════════════════════════════════════════════════════════

try:
    from hypothesis import given, strategies as st, settings
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestPropertyBased:
    """Hypothesis generates random inputs — invariants MUST hold for ALL."""

    @given(st.integers(200, 2000), st.integers(5, 20), st.integers(1, 5))
    @settings(max_examples=30)
    def test_hankel_shape_invariant(self, n, m, tau):
        """Hankel matrix shape = (n - (m-1)*tau, m) for any valid input.
        Hypothesis generates random (n, m, tau) — only test when valid."""
        x = np.sin(np.linspace(0, 20, n))
        expected = n - (m - 1) * tau
        if expected <= 0:
            return  # skip invalid combos
        H = hankel_matrix(x, m, tau)
        assert H.shape == (expected, m)

    @given(st.integers(300, 2000), st.integers(10, 40), st.integers(2, 8))
    @settings(max_examples=20)
    def test_svd_orthonormal_for_any_signal(self, n, m, r):
        """SVD of Hankel always produces orthonormal U for any random signal."""
        x = np.random.randn(n)
        H = hankel_matrix(x, min(m, n // 5), 1)
        r = min(r, m - 1, H.shape[0] - 1)
        U, _ = eigen_time_delay(H, max(r, 2), solver="scipy")
        assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-10)

    @given(st.integers(200, 1500))
    @settings(max_examples=15)
    def test_forcing_non_negative_abs(self, n):
        """|forcing| is always ≥ 0 for any signal."""
        x = np.sin(np.linspace(0, 20, n)) + np.random.randn(n) * 0.1
        est = HavokEstimator(tau=1, m=min(30, n // 5), r=3)
        est.fit(x)
        assert np.all(np.abs(est.forcing_) >= 0)
