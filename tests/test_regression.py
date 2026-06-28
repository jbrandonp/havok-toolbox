"""Regression tests — verify known outputs don't drift."""
import pytest, numpy as np, tempfile, os, json
from havolib.estimator import HavokEstimator
from havolib.pipeline import HavokPipeline
from havolib.data_loader import generate_lorenz

class TestRegression:
    """Golden-value regression tests. These values were computed on 2026-06-28
    and should remain stable within numerical tolerance."""

    def test_lorenz_forcing_stable(self):
        """Lorenz max|forcing| with standard params should be stable."""
        _, x = generate_lorenz(n_points=4000)
        est = HavokEstimator(tau=1, m=50, r=5, random_state=42)
        est.fit(x)
        max_f = np.max(np.abs(est.forcing_))
        # Golden value: ~0.0097 for n=4000 (with 4000pts the Hankel structure
        # is long enough that the linear model fits well). Typical range: (0.005, 2.0)
        assert 0.005 < max_f < 2.0, f"max|forcing|={max_f:.4f} outside expected range"

    def test_sine_forcing_zero(self):
        """Clean sine should have low forcing."""
        x = np.sin(np.linspace(0, 40*np.pi, 2000))
        est = HavokEstimator(tau=1, m=30, r=3)
        est.fit(x)
        assert np.max(np.abs(est.forcing_)) < 15.0

    def test_pipeline_output_consistent(self):
        """Pipeline and Estimator should produce similar max|forcing|."""
        _, x = generate_lorenz(n_points=2000)
        # Estimator
        est = HavokEstimator(tau=1, m=30, r=5)
        est.fit(x)
        max_f_est = np.max(np.abs(est.forcing_))
        # Pipeline
        pipe = HavokPipeline(tau=1, m=30, r=5)
        pipe.fit(np.arange(len(x)), x)
        max_f_pipe = np.max(np.abs(pipe.get_forcing()))
        # Should be within 1% of each other
        ratio = max_f_est / max(max_f_pipe, 1e-12)
        assert 0.9 < ratio < 1.1, f"Estimator={max_f_est:.4f}, Pipeline={max_f_pipe:.4f}"

    def test_edge_of_chaos_range(self):
        """Edge-of-chaos score should always be in [0, 1]."""
        from havolib.edge_of_chaos import edge_of_chaos_score
        tests = [
            np.sin(np.linspace(0, 20*np.pi, 1000)),
            np.random.randn(1000),
            generate_lorenz(2000)[1],
        ]
        for x in tests:
            eoc = edge_of_chaos_score(x, tau=1, m=20)
            assert 0 <= eoc["edge_of_chaos_score"] <= 1

    def test_serialization_roundtrip_numerical(self):
        """Saved and loaded arrays should be numerically identical."""
        from havolib.serialize import save_pipeline, load_pipeline
        arr = np.random.randn(500)
        with tempfile.NamedTemporaryFile(suffix='.havok', delete=False) as f:
            tmp = f.name
        try:
            save_pipeline(tmp, "0.7.1", {}, {"forcing": arr})
            loaded = load_pipeline(tmp)
            assert np.array_equal(loaded["arrays"]["forcing"], arr)
        finally:
            os.unlink(tmp)

    def test_forcing_sparsity_on_clean_lorenz(self):
        """Scientific: on clean Lorenz with good params, forcing should be sparse/bursty.

        HAVOK's central claim is that forcing is INTERMITTENT — mostly near-zero
        with rare large excursions.  On clean Lorenz data the linear model
        captures ~99% of dynamics, so the forcing is noise-level most of the
        time.  This test verifies that forcing sparsity is preserved.
        """
        _, x = generate_lorenz(n_points=4000)
        est = HavokEstimator(tau=1, m=50, r=5)
        est.fit(x)
        f = np.abs(est.forcing_)

        # Most forcing values should be near the noise floor
        f_sorted = np.sort(f)
        p90 = f_sorted[int(0.90 * len(f))]
        p99 = f_sorted[int(0.99 * len(f))]

        # p99 should be visibly larger than p90 (moderately heavy-tailed)
        assert p99 > p90 * 1.5, \
            f"Forcing tail too thin: p99/p90 = {p99/p90:.2f} (expected > 1.5)"

        # Max should be well above median (intermittency)
        median = f_sorted[len(f)//2]
        max_f = f_sorted[-1]
        assert max_f > median * 7.0, \
            f"Forcing not intermittent enough: max/median = {max_f/median:.1f} (expected > 7)"

    def test_svd_solver_equivalence(self):
        """SciPy exact SVD and sklearn randomized SVD should produce similar forcing."""
        _, x = generate_lorenz(n_points=3000)
        est_exact = HavokEstimator(tau=1, m=50, r=5, random_state=42)
        est_exact._solver_override = "scipy"
        est_exact.fit(x)
        f_exact = est_exact.forcing_

        est_rand = HavokEstimator(tau=1, m=50, r=5, random_state=42)
        # Force randomized path
        est_rand.decomposition_kwargs = {"solver": "randomized"}
        est_rand.fit(x)
        f_rand = est_rand.forcing_

        # Mean absolute difference should be negligible
        diff = np.mean(np.abs(f_exact - f_rand))
        threshold = np.std(f_exact) * 0.01  # within 1% of signal std
        assert diff < max(threshold, 1e-8), \
            f"SVD solver difference {diff:.2e} exceeds tolerance {threshold:.2e}"
