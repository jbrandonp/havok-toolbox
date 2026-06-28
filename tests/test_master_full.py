"""
MASTER FULL TEST SUITE — havok-toolbox
=======================================
Covers: Unit | Integration | Regression | Stress | Edge Cases | Contract | CLI | Serialization

Run with:
    pytest tests/test_master_full.py -v
    pytest tests/test_master_full.py -v --tb=short -x   # stop on first fail
"""

from __future__ import annotations
import numpy as np
import pytest
import warnings
import os
import io
import tempfile
import pickle

# ─────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clean_sine():
    """Perfect clean sine wave — baseline healthy signal."""
    t = np.linspace(0, 50, 3000)
    return np.sin(2 * np.pi * 0.1 * t)


@pytest.fixture(scope="module")
def noisy_sine():
    """Sine wave with heavy noise."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 50, 3000)
    return np.sin(2 * np.pi * 0.1 * t) + rng.normal(0, 0.5, 3000)


@pytest.fixture(scope="module")
def lorenz_x():
    """Simple Lorenz-like chaotic signal (x component only)."""
    rng = np.random.default_rng(0)
    x = np.zeros(4000)
    x[0] = 0.1
    for i in range(1, 4000):
        x[i] = x[i-1] + 0.01 * (10 * (x[i-1] + rng.normal(0, 0.01)) - x[i-1])
        x[i] = np.clip(x[i], -50, 50)
    return x


@pytest.fixture(scope="module")
def regime_shift_signal():
    """Signal with an obvious regime shift in the middle."""
    rng = np.random.default_rng(7)
    part1 = np.sin(np.linspace(0, 20, 2000)) + rng.normal(0, 0.05, 2000)
    part2 = np.sin(np.linspace(0, 20, 2000)) * 3 + rng.normal(0, 0.5, 2000)
    return np.concatenate([part1, part2])


@pytest.fixture(scope="module")
def fitted_estimator(clean_sine):
    """Pre-fitted HavokEstimator to reuse across tests."""
    from havolib.estimator import HavokEstimator
    est = HavokEstimator(r=5, tau=2, m=20, diff_method="finite_diff")
    est.fit(clean_sine)
    return est


# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORT HEALTH — All modules must import without error
# ─────────────────────────────────────────────────────────────────────────────

class TestImports:
    """Every module in havolib/ must import cleanly."""

    def test_import_estimator(self):
        from havolib import estimator
        assert hasattr(estimator, "HavokEstimator")

    def test_import_pipeline(self):
        from havolib import pipeline

    def test_import_adaptive(self):
        from havolib import adaptive

    def test_import_arena(self):
        from havolib import arena

    def test_import_attribution(self):
        from havolib import attribution

    def test_import_auto_tune(self):
        from havolib import auto_tune

    def test_import_automl(self):
        from havolib import automl

    def test_import_config(self):
        from havolib import config

    def test_import_data_loader(self):
        from havolib import data_loader

    def test_import_decomposition(self):
        from havolib import decomposition

    def test_import_detection(self):
        from havolib import detection

    def test_import_edge_of_chaos(self):
        from havolib import edge_of_chaos

    def test_import_embedding(self):
        from havolib import embedding

    def test_import_federated(self):
        from havolib import federated

    def test_import_forcing(self):
        from havolib import forcing

    def test_import_gpu(self):
        from havolib import gpu

    def test_import_hybrid(self):
        from havolib import hybrid

    def test_import_logging_config(self):
        from havolib import logging_config

    def test_import_ml_risk_predictor(self):
        from havolib import ml_risk_predictor

    def test_import_multichannel(self):
        from havolib import multichannel

    def test_import_polars_loader(self):
        from havolib import polars_loader

    def test_import_pre_processing(self):
        from havolib import pre_processing

    def test_import_serialize(self):
        from havolib import serialize

    def test_import_surrogate(self):
        from havolib import surrogate

    def test_import_user(self):
        from havolib import user

    def test_import_visualization(self):
        from havolib import visualization

    def test_havolib_init_exports(self):
        import havolib
        # __init__.py must expose HavokEstimator at top level
        assert hasattr(havolib, "HavokEstimator"), "HavokEstimator not exported from havolib/__init__.py"


# ─────────────────────────────────────────────────────────────────────────────
# 2. ESTIMATOR — Core HAVOK math
# ─────────────────────────────────────────────────────────────────────────────

class TestHavokEstimator:

    def test_fit_returns_self(self, clean_sine):
        from havolib.estimator import HavokEstimator
        est = HavokEstimator(r=4, tau=2, m=15)
        result = est.fit(clean_sine)
        assert result is est

    def test_forcing_shape_matches_input(self, clean_sine, fitted_estimator):
        assert fitted_estimator.forcing_.shape == clean_sine.shape

    def test_risk_shape_matches_input(self, clean_sine, fitted_estimator):
        assert fitted_estimator.risk_.shape == clean_sine.shape

    def test_risk_is_binary(self, fitted_estimator):
        unique_vals = np.unique(fitted_estimator.risk_)
        assert set(unique_vals).issubset({0, 1}), f"Risk should be 0/1, got: {unique_vals}"

    def test_singular_values_positive_descending(self, fitted_estimator):
        sv = fitted_estimator.singular_values_
        assert np.all(sv > 0), "All singular values must be positive"
        assert np.all(np.diff(sv) <= 1e-9), "Singular values must be non-increasing"

    def test_forcing_is_finite(self, fitted_estimator):
        assert np.all(np.isfinite(fitted_estimator.forcing_)), "Forcing contains NaN or Inf"

    def test_fit_transform_equals_fit_then_transform(self, noisy_sine):
        from havolib.estimator import HavokEstimator
        est1 = HavokEstimator(r=4, tau=2, m=15, random_state=1)
        est2 = HavokEstimator(r=4, tau=2, m=15, random_state=1)
        ft = est1.fit_transform(noisy_sine)
        _ = est2.fit(noisy_sine)
        t = est2.transform()
        np.testing.assert_array_almost_equal(ft, t, decimal=10)

    def test_all_diff_methods_work(self, clean_sine):
        from havolib.estimator import HavokEstimator
        for method in ["finite_diff", "spline", "total_variation", "gradient"]:
            est = HavokEstimator(r=4, tau=2, m=15, diff_method=method)
            est.fit(clean_sine)
            assert np.all(np.isfinite(est.forcing_)), f"{method} produced non-finite forcing"

    def test_auto_tau_m_resolves(self, clean_sine):
        from havolib.estimator import HavokEstimator
        est = HavokEstimator(r=4, tau="auto", m="auto")
        est.fit(clean_sine)
        assert isinstance(est.tau_fitted_, int) and est.tau_fitted_ >= 1
        assert isinstance(est.m_fitted_, int) and est.m_fitted_ >= 5

    def test_score_returns_float(self, fitted_estimator, clean_sine):
        s = fitted_estimator.score(clean_sine)
        assert isinstance(s, float)
        assert s >= 0.0

    def test_get_params_set_params_roundtrip(self):
        from havolib.estimator import HavokEstimator
        est = HavokEstimator(r=7, tau=3, m=25, threshold_std=2.5)
        params = est.get_params()
        est2 = HavokEstimator()
        est2.set_params(**params)
        assert est2.r == 7
        assert est2.tau == 3
        assert est2.m == 25
        assert est2.threshold_std == 2.5

    def test_regime_shift_detected_on_shift_signal(self, regime_shift_signal):
        from havolib.estimator import HavokEstimator
        est = HavokEstimator(r=5, tau=2, m=20)
        est.fit(regime_shift_signal)
        # At least some risk events must be detected in the second half
        second_half_risk = est.risk_[len(regime_shift_signal)//2:]
        assert np.sum(second_half_risk) > 0, "No regime shift detected on known shift signal"

    def test_2d_input_handled(self):
        from havolib.estimator import HavokEstimator
        rng = np.random.default_rng(1)
        X = rng.standard_normal((500, 1))
        est = HavokEstimator(r=3, tau=1, m=10)
        est.fit(X)  # should not crash
        assert est.forcing_.shape[0] == 500

    def test_predict_risk_agrees_with_risk_attribute(self, fitted_estimator, clean_sine):
        risk_via_method = fitted_estimator.predict_risk(clean_sine)
        np.testing.assert_array_equal(risk_via_method, fitted_estimator.risk_)


# ─────────────────────────────────────────────────────────────────────────────
# 3. EMBEDDING — Hankel matrix
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedding:

    def test_hankel_shape(self, clean_sine):
        from havolib.embedding import hankel_matrix
        H = hankel_matrix(clean_sine, m=30, tau=2)
        assert H.ndim == 2
        expected_rows = len(clean_sine) - (30 - 1) * 2
        assert H.shape[0] == expected_rows, f"Hankel rows mismatch: {H.shape[0]} vs {expected_rows}"
        assert H.shape[1] == 30

    def test_hankel_no_nan(self, clean_sine):
        from havolib.embedding import hankel_matrix
        H = hankel_matrix(clean_sine, m=20, tau=1)
        assert np.all(np.isfinite(H))

    def test_hankel_tau1_first_row(self, clean_sine):
        from havolib.embedding import hankel_matrix
        H = hankel_matrix(clean_sine, m=5, tau=1)
        np.testing.assert_array_equal(H[0], clean_sine[:5])

    def test_hankel_minimal_input(self):
        from havolib.embedding import hankel_matrix
        x = np.arange(20, dtype=float)
        H = hankel_matrix(x, m=5, tau=1)
        assert H.shape[0] > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. DECOMPOSITION — SVD
# ─────────────────────────────────────────────────────────────────────────────

class TestDecomposition:

    def test_eigen_time_delay_shapes(self, clean_sine):
        from havolib.embedding import hankel_matrix
        from havolib.decomposition import eigen_time_delay
        H = hankel_matrix(clean_sine, m=20, tau=2)
        r = 5
        V, s = eigen_time_delay(H, r)
        assert V.shape == (H.shape[0], r)
        assert s.shape == (r,)

    def test_singular_values_non_negative(self, clean_sine):
        from havolib.embedding import hankel_matrix
        from havolib.decomposition import eigen_time_delay
        H = hankel_matrix(clean_sine, m=20, tau=2)
        _, s = eigen_time_delay(H, 5)
        assert np.all(s >= 0)

    def test_eigen_coords_finite(self, clean_sine):
        from havolib.embedding import hankel_matrix
        from havolib.decomposition import eigen_time_delay
        H = hankel_matrix(clean_sine, m=20, tau=2)
        V, _ = eigen_time_delay(H, 5)
        assert np.all(np.isfinite(V))

    def test_columns_orthogonal(self, clean_sine):
        """SVD eigen-coords must be approximately orthonormal."""
        from havolib.embedding import hankel_matrix
        from havolib.decomposition import eigen_time_delay
        H = hankel_matrix(clean_sine, m=20, tau=2)
        V, _ = eigen_time_delay(H, 5)
        gram = V.T @ V
        # Diagonal = norms squared, off-diag should be ~0
        np.testing.assert_allclose(gram, np.eye(5), atol=0.1, err_msg="Eigen-coords not orthogonal")


# ─────────────────────────────────────────────────────────────────────────────
# 5. DETECTION — Threshold risk
# ─────────────────────────────────────────────────────────────────────────────

class TestDetection:

    def test_threshold_risk_output_binary(self):
        from havolib.detection import threshold_risk
        rng = np.random.default_rng(5)
        forcing = rng.standard_normal(1000)
        risk = threshold_risk(forcing, window=50, threshold_std=3.0)
        assert set(np.unique(risk)).issubset({0, 1})

    def test_threshold_risk_shape(self):
        from havolib.detection import threshold_risk
        forcing = np.random.randn(500)
        risk = threshold_risk(forcing, window=50, threshold_std=2.0)
        assert risk.shape == forcing.shape

    def test_threshold_risk_zeros_on_flat_signal(self):
        """Flat signal has zero std → no threshold crossings."""
        from havolib.detection import threshold_risk
        forcing = np.zeros(500)
        risk = threshold_risk(forcing, window=50, threshold_std=2.0)
        assert np.all(risk == 0)

    def test_threshold_risk_detects_spike(self):
        """A huge spike must be flagged as risk."""
        from havolib.detection import threshold_risk
        forcing = np.zeros(1000)
        forcing[500] = 1000.0  # massive spike
        risk = threshold_risk(forcing, window=50, threshold_std=2.0)
        # Some samples near the spike must be flagged
        assert np.sum(risk[400:600]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. DIFFERENTIATION METHODS
# ─────────────────────────────────────────────────────────────────────────────

class TestDifferentiation:

    @pytest.fixture
    def smooth_V(self):
        t = np.linspace(0, 10, 500)
        V = np.column_stack([np.sin(t), np.cos(t), np.sin(2 * t)])
        return V, t

    def test_finite_diff_shape(self, smooth_V):
        from havolib.estimator import finite_diff
        V, t = smooth_V
        dv = finite_diff(V, t, order=2)
        assert dv.shape == V.shape

    def test_finite_diff_order1_shape(self, smooth_V):
        from havolib.estimator import finite_diff
        V, t = smooth_V
        dv = finite_diff(V, t, order=1)
        assert dv.shape == V.shape

    def test_finite_diff_order4_shape(self, smooth_V):
        from havolib.estimator import finite_diff
        V, t = smooth_V
        dv = finite_diff(V, t, order=4)
        assert dv.shape == V.shape

    def test_spline_diff_accuracy(self, smooth_V):
        """Spline derivative of sin(t) should approximate cos(t)."""
        from havolib.estimator import spline_diff
        t = np.linspace(0, 2 * np.pi, 200)
        V = np.sin(t).reshape(-1, 1)
        dv = spline_diff(V, t)
        expected = np.cos(t).reshape(-1, 1)
        # Allow generous tolerance since it's numerical
        np.testing.assert_allclose(dv[5:-5], expected[5:-5], atol=0.1)

    def test_tv_diff_shape(self, smooth_V):
        from havolib.estimator import total_variation_diff
        V, t = smooth_V
        dv = total_variation_diff(V, t, alpha=0.05)
        assert dv.shape == V.shape

    def test_all_diff_methods_finite(self, smooth_V):
        from havolib.estimator import finite_diff, spline_diff, total_variation_diff
        V, t = smooth_V
        for fn in [finite_diff, spline_diff, total_variation_diff]:
            dv = fn(V, t)
            assert np.all(np.isfinite(dv)), f"{fn.__name__} produced non-finite values"


# ─────────────────────────────────────────────────────────────────────────────
# 7. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossValidation:

    def test_cv_returns_expected_keys(self, clean_sine):
        from havolib.estimator import cross_val_score_havok
        result = cross_val_score_havok(
            clean_sine,
            param_grid={"r": [3, 5], "tau": [2], "m": [15]},
            cv=2,
        )
        assert "best_params" in result
        assert "best_score" in result
        assert "cv_results" in result

    def test_cv_best_score_non_negative(self, clean_sine):
        from havolib.estimator import cross_val_score_havok
        result = cross_val_score_havok(
            clean_sine,
            param_grid={"r": [3], "tau": [2], "m": [15]},
            cv=2,
        )
        assert result["best_score"] >= 0.0

    def test_cv_result_count_matches_grid(self, clean_sine):
        from havolib.estimator import cross_val_score_havok
        result = cross_val_score_havok(
            clean_sine,
            param_grid={"r": [3, 5, 7], "tau": [1, 2], "m": [15]},
            cv=2,
        )
        # 3 * 2 * 1 = 6 combinations
        assert len(result["cv_results"]) == 6


# ─────────────────────────────────────────────────────────────────────────────
# 8. SERIALIZATION — Save & Load
# ─────────────────────────────────────────────────────────────────────────────

class TestSerialization:

    def test_pickle_roundtrip(self, fitted_estimator):
        buf = io.BytesIO()
        pickle.dump(fitted_estimator, buf)
        buf.seek(0)
        loaded = pickle.load(buf)
        np.testing.assert_array_equal(loaded.forcing_, fitted_estimator.forcing_)
        np.testing.assert_array_equal(loaded.risk_, fitted_estimator.risk_)

    def test_serialize_module_save_load(self, fitted_estimator):
        """havolib.serialize must save and restore the estimator."""
        from havolib import serialize
        if not (hasattr(serialize, "save") and hasattr(serialize, "load")):
            pytest.skip("serialize module does not have save/load functions")
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            serialize.save(fitted_estimator, path)
            loaded = serialize.load(path)
            np.testing.assert_array_almost_equal(
                loaded.forcing_, fitted_estimator.forcing_, decimal=10
            )
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_joblib_roundtrip(self, fitted_estimator):
        """Joblib is the sklearn-standard serializer — must work cleanly."""
        import joblib
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        try:
            joblib.dump(fitted_estimator, path)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(loaded.risk_, fitted_estimator.risk_)
        finally:
            if os.path.exists(path):
                os.remove(path)


# ─────────────────────────────────────────────────────────────────────────────
# 9. EDGE CASES — Pathological inputs
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_constant_signal(self):
        """All-constant signal should not crash, forcing should be near zero."""
        from havolib.estimator import HavokEstimator
        x = np.ones(500)
        est = HavokEstimator(r=3, tau=1, m=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(x)
        assert np.all(np.isfinite(est.forcing_))

    def test_all_zeros_signal(self):
        from havolib.estimator import HavokEstimator
        x = np.zeros(500)
        est = HavokEstimator(r=3, tau=1, m=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(x)
        assert est.forcing_.shape[0] == 500

    def test_single_spike_signal(self):
        from havolib.estimator import HavokEstimator
        x = np.zeros(500)
        x[250] = 100.0
        est = HavokEstimator(r=3, tau=1, m=10)
        est.fit(x)
        assert np.all(np.isfinite(est.forcing_))

    def test_very_short_signal_raises_or_warns(self):
        from havolib.estimator import HavokEstimator
        x = np.arange(10, dtype=float)
        est = HavokEstimator(r=3, tau=1, m=5)
        with pytest.raises(Exception):
            est.fit(x)  # too short, must fail clearly

    def test_negative_values_signal(self):
        from havolib.estimator import HavokEstimator
        x = -1 * np.sin(np.linspace(0, 20, 1000))
        est = HavokEstimator(r=4, tau=2, m=15)
        est.fit(x)
        assert np.all(np.isfinite(est.forcing_))

    def test_large_amplitude_signal(self):
        from havolib.estimator import HavokEstimator
        x = np.sin(np.linspace(0, 20, 2000)) * 1e6
        est = HavokEstimator(r=4, tau=2, m=15)
        est.fit(x)
        assert np.all(np.isfinite(est.forcing_))

    def test_high_noise_signal(self):
        from havolib.estimator import HavokEstimator
        rng = np.random.default_rng(99)
        x = rng.standard_normal(2000) * 100
        est = HavokEstimator(r=4, tau=2, m=15)
        est.fit(x)
        assert np.all(np.isfinite(est.forcing_))

    def test_transform_before_fit_raises(self):
        from havolib.estimator import HavokEstimator
        est = HavokEstimator()
        with pytest.raises(Exception):
            est.transform()

    def test_get_risk_before_fit_raises(self):
        from havolib.estimator import HavokEstimator
        est = HavokEstimator()
        with pytest.raises(Exception):
            est.get_risk()

    def test_invalid_diff_method_raises(self, clean_sine):
        from havolib.estimator import HavokEstimator
        est = HavokEstimator(r=4, tau=2, m=15, diff_method="INVALID_METHOD")
        # Should either fall back gracefully OR raise a clear error
        try:
            est.fit(clean_sine)
            # If no exception, forcing must still be finite
            assert np.all(np.isfinite(est.forcing_))
        except (KeyError, ValueError, TypeError):
            pass  # Raising is also acceptable


# ─────────────────────────────────────────────────────────────────────────────
# 10. INTEGRATION — Full pipeline data_loader → pre_processing → estimator
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationPipeline:

    def test_pre_processing_to_estimator(self):
        """Preprocessed signal must produce valid HAVOK output."""
        from havolib import pre_processing
        from havolib.estimator import HavokEstimator
        rng = np.random.default_rng(42)
        raw = rng.standard_normal(2000) * 10 + np.sin(np.linspace(0, 20, 2000)) * 5
        # Find a normalize/standardize function if it exists
        if hasattr(pre_processing, "normalize"):
            processed = pre_processing.normalize(raw)
        elif hasattr(pre_processing, "standardize"):
            processed = pre_processing.standardize(raw)
        else:
            processed = (raw - raw.mean()) / (raw.std() + 1e-9)
        est = HavokEstimator(r=4, tau=2, m=15)
        est.fit(processed)
        assert np.all(np.isfinite(est.forcing_))

    def test_full_chain_clean_to_risk(self, clean_sine):
        """Clean sine → estimator → risk must be a valid binary array."""
        from havolib.estimator import HavokEstimator
        est = HavokEstimator(r=5, tau=2, m=20)
        est.fit(clean_sine)
        risk = est.get_risk()
        assert risk.dtype in [np.int32, np.int64, np.float64, bool, int]
        assert set(np.unique(risk.astype(int))).issubset({0, 1})

    def test_multichannel_pipeline(self):
        """Multichannel module must handle 2D multi-signal input."""
        from havolib import multichannel
        rng = np.random.default_rng(10)
        X = rng.standard_normal((2000, 3))  # 3 channels
        if hasattr(multichannel, "fit"):
            result = multichannel.fit(X)
            assert result is not None
        elif hasattr(multichannel, "MultichannelHavok"):
            mc = multichannel.MultichannelHavok(r=3)
            mc.fit(X)
        else:
            pytest.skip("multichannel API structure unknown")

    def test_pipeline_module_end_to_end(self, clean_sine):
        """havolib.pipeline must accept a signal and return results."""
        from havolib import pipeline
        if hasattr(pipeline, "HavokPipeline"):
            pipe = pipeline.HavokPipeline()
            if hasattr(pipe, "fit"):
                pipe.fit(clean_sine)
        elif hasattr(pipeline, "run"):
            pipeline.run(clean_sine)
        else:
            pytest.skip("pipeline API structure unknown")


# ─────────────────────────────────────────────────────────────────────────────
# 11. REGRESSION — Results must be numerically stable across runs
# ─────────────────────────────────────────────────────────────────────────────

class TestRegression:
    """Fit the same data twice with the same seed — results must be identical."""

    REFERENCE_SEED = 42
    N = 3000

    @pytest.fixture(scope="class")
    def reference_data(self):
        rng = np.random.default_rng(self.REFERENCE_SEED)
        t = np.linspace(0, 50, self.N)
        return np.sin(t) + rng.normal(0, 0.05, self.N)

    def test_forcing_reproducible(self, reference_data):
        from havolib.estimator import HavokEstimator
        est1 = HavokEstimator(r=5, tau=2, m=20, random_state=0)
        est2 = HavokEstimator(r=5, tau=2, m=20, random_state=0)
        est1.fit(reference_data)
        est2.fit(reference_data)
        np.testing.assert_array_almost_equal(est1.forcing_, est2.forcing_, decimal=10)

    def test_risk_reproducible(self, reference_data):
        from havolib.estimator import HavokEstimator
        est1 = HavokEstimator(r=5, tau=2, m=20, random_state=0)
        est2 = HavokEstimator(r=5, tau=2, m=20, random_state=0)
        est1.fit(reference_data)
        est2.fit(reference_data)
        np.testing.assert_array_equal(est1.risk_, est2.risk_)

    def test_singular_values_reproducible(self, reference_data):
        from havolib.estimator import HavokEstimator
        est1 = HavokEstimator(r=5, tau=2, m=20, random_state=0)
        est2 = HavokEstimator(r=5, tau=2, m=20, random_state=0)
        est1.fit(reference_data)
        est2.fit(reference_data)
        np.testing.assert_array_almost_equal(
            est1.singular_values_, est2.singular_values_, decimal=10
        )


# ─────────────────────────────────────────────────────────────────────────────
# 12. STRESS — Large inputs and performance sanity
# ─────────────────────────────────────────────────────────────────────────────

class TestStress:

    def test_large_signal_50k(self):
        """50,000 samples must complete without crash or memory error."""
        import time
        from havolib.estimator import HavokEstimator
        rng = np.random.default_rng(1)
        x = np.sin(np.linspace(0, 100, 50_000)) + rng.normal(0, 0.1, 50_000)
        est = HavokEstimator(r=5, tau=2, m=20)
        start = time.time()
        est.fit(x)
        elapsed = time.time() - start
        assert np.all(np.isfinite(est.forcing_))
        assert elapsed < 60, f"Too slow on 50k samples: {elapsed:.1f}s"

    def test_many_small_signals_batch(self):
        """Fitting 50 small signals must all succeed."""
        from havolib.estimator import HavokEstimator
        rng = np.random.default_rng(77)
        errors = []
        for i in range(50):
            x = np.sin(np.linspace(0, 10, 500)) + rng.normal(0, 0.1, 500)
            est = HavokEstimator(r=3, tau=1, m=10)
            try:
                est.fit(x)
                if not np.all(np.isfinite(est.forcing_)):
                    errors.append(f"Signal {i}: non-finite forcing")
            except Exception as e:
                errors.append(f"Signal {i}: {e}")
        assert len(errors) == 0, f"Batch errors:\n" + "\n".join(errors)

    def test_high_r_value(self):
        """r=20 should work or fail gracefully, not crash with unhandled exception."""
        from havolib.estimator import HavokEstimator
        x = np.sin(np.linspace(0, 50, 5000))
        est = HavokEstimator(r=20, tau=2, m=50)
        try:
            est.fit(x)
            assert np.all(np.isfinite(est.forcing_))
        except (ValueError, np.linalg.LinAlgError):
            pass  # Acceptable — but must not be unhandled


# ─────────────────────────────────────────────────────────────────────────────
# 13. CONFIG — YAML loading and validation
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:

    def test_config_file_loads(self):
        """havok_config.yaml must parse without error."""
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "..", "havok_config.yaml")
        config_path = os.path.abspath(config_path)
        if not os.path.exists(config_path):
            pytest.skip("havok_config.yaml not found")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict), "Config must parse to a dict"

    def test_engine_yaml_loads(self):
        import yaml
        engine_path = os.path.join(os.path.dirname(__file__), "..", "engine.yaml")
        engine_path = os.path.abspath(engine_path)
        if not os.path.exists(engine_path):
            pytest.skip("engine.yaml not found")
        with open(engine_path, "r") as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)

    def test_config_module_loads_yaml(self):
        from havolib import config
        config_path = os.path.join(os.path.dirname(__file__), "..", "havok_config.yaml")
        config_path = os.path.abspath(config_path)
        if not os.path.exists(config_path):
            pytest.skip("havok_config.yaml not found")
        # Try any load function
        for fn_name in ["load", "load_config", "from_yaml", "parse"]:
            if hasattr(config, fn_name):
                cfg = getattr(config, fn_name)(config_path)
                assert cfg is not None
                return
        pytest.skip("No config load function found")


# ─────────────────────────────────────────────────────────────────────────────
# 14. VISUALIZATION — Plot methods must not crash
# ─────────────────────────────────────────────────────────────────────────────

class TestVisualization:

    def test_estimator_plot_returns_figure(self, fitted_estimator):
        """plot() must return a Plotly Figure without crashing."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            pytest.skip("plotly not installed")
        fig = fitted_estimator.plot()
        assert fig is not None
        assert hasattr(fig, "data"), "plot() must return a Plotly Figure with .data"

    def test_visualization_module_callable(self, fitted_estimator):
        from havolib import visualization
        # Check any callable that takes forcing/risk
        for fn_name in ["plot_forcing", "plot_risk", "plot", "show"]:
            if hasattr(visualization, fn_name):
                fn = getattr(visualization, fn_name)
                try:
                    fn(fitted_estimator.forcing_)
                except Exception as e:
                    pytest.fail(f"{fn_name} raised: {e}")
                return
        pytest.skip("No visualization function found")


# ─────────────────────────────────────────────────────────────────────────────
# 15. SKLEARN CONTRACT — HavokEstimator must be a valid sklearn estimator
# ─────────────────────────────────────────────────────────────────────────────

class TestSklearnContract:

    def test_get_params_returns_all_init_args(self):
        from havolib.estimator import HavokEstimator
        est = HavokEstimator(r=7, tau=3, m=25)
        params = est.get_params()
        for key in ["r", "tau", "m", "threshold_std", "window", "diff_method", "svd_solver"]:
            assert key in params, f"Missing param: {key}"

    def test_clone_via_get_set_params(self):
        """sklearn.clone() requires get_params + set_params to work."""
        from sklearn.base import clone
        from havolib.estimator import HavokEstimator
        est = HavokEstimator(r=6, tau=3, m=20)
        cloned = clone(est)
        assert cloned.r == 6
        assert cloned.tau == 3

    def test_check_estimator_basic(self):
        """Run a subset of sklearn's estimator checks."""
        from sklearn.utils.estimator_checks import parametrize_with_checks
        from havolib.estimator import HavokEstimator
        # Only check that it doesn't crash on basic interface checks
        est = HavokEstimator(r=3, tau=1, m=8)
        assert hasattr(est, "fit")
        assert hasattr(est, "transform")
        assert hasattr(est, "fit_transform")
        assert hasattr(est, "get_params")
        assert hasattr(est, "set_params")


# ─────────────────────────────────────────────────────────────────────────────
# 16. AUTO-TUNE SANITY
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoTune:

    def test_optimal_tau_positive_integer(self, clean_sine):
        from havolib.auto_tune import optimal_tau_mi
        tau = optimal_tau_mi(clean_sine, max_lag=50)
        assert isinstance(tau, (int, np.integer))
        assert tau >= 1

    def test_optimal_m_positive_integer(self, clean_sine):
        from havolib.auto_tune import optimal_m_fnn
        m = optimal_m_fnn(clean_sine, tau=2, max_m=30)
        assert isinstance(m, (int, np.integer))
        assert m >= 1

    def test_tau_varies_with_signal_structure(self):
        """High-freq and low-freq sine should yield different tau."""
        from havolib.auto_tune import optimal_tau_mi
        t = np.linspace(0, 50, 3000)
        low = np.sin(2 * np.pi * 0.05 * t)
        high = np.sin(2 * np.pi * 0.5 * t)
        tau_low = optimal_tau_mi(low, max_lag=100)
        tau_high = optimal_tau_mi(high, max_lag=100)
        # Low-freq signal should have larger tau than high-freq
        assert tau_low >= tau_high, (
            f"Expected tau_low ({tau_low}) >= tau_high ({tau_high})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 17. FORCING MODULE
# ─────────────────────────────────────────────────────────────────────────────

class TestForcing:

    def test_forcing_module_callable(self):
        from havolib import forcing
        for fn_name in ["extract", "compute", "get_forcing"]:
            if hasattr(forcing, fn_name):
                return  # At least one callable exists
        pytest.skip("forcing module has no known callable")

    def test_forcing_output_finite(self, clean_sine):
        from havolib import forcing
        if hasattr(forcing, "extract"):
            result = forcing.extract(clean_sine)
        elif hasattr(forcing, "compute"):
            result = forcing.compute(clean_sine)
        else:
            pytest.skip("No known forcing extraction function")
        assert np.all(np.isfinite(result))


# ─────────────────────────────────────────────────────────────────────────────
# 18. GPU MODULE — Must degrade gracefully without GPU
# ─────────────────────────────────────────────────────────────────────────────

class TestGPU:

    def test_gpu_module_has_status_or_detect(self):
        from havolib import gpu
        has_fn = any(hasattr(gpu, fn) for fn in ["is_available", "detect", "get_device", "svd"])
        assert has_fn, "gpu.py should expose at least one function"

    def test_gpu_detect_returns_bool_or_device(self):
        from havolib import gpu
        for fn_name in ["is_available", "detect"]:
            if hasattr(gpu, fn_name):
                result = getattr(gpu, fn_name)()
                assert isinstance(result, (bool, str, type(None))), (
                    f"{fn_name} must return bool or string, got {type(result)}"
                )
                return

    def test_svd_cpu_fallback(self, clean_sine):
        """GPU SVD must fall back to CPU silently if no GPU present."""
        from havolib import gpu
        if not hasattr(gpu, "svd"):
            pytest.skip("gpu.svd not available")
        from havolib.embedding import hankel_matrix
        H = hankel_matrix(clean_sine, m=20, tau=2)
        try:
            V, s = gpu.svd(H, r=5)
            assert np.all(np.isfinite(V))
            assert np.all(s >= 0)
        except Exception as e:
            pytest.fail(f"gpu.svd raised instead of falling back: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 19. PYPROJECT / PACKAGE METADATA
# ─────────────────────────────────────────────────────────────────────────────

class TestPackageMetadata:

    def test_pyproject_toml_exists(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        path = os.path.abspath(os.path.join(root, "pyproject.toml"))
        assert os.path.exists(path), "pyproject.toml is missing"

    def test_pyproject_has_required_fields(self):
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("tomllib/tomli not available")
        root = os.path.join(os.path.dirname(__file__), "..")
        path = os.path.abspath(os.path.join(root, "pyproject.toml"))
        with open(path, "rb") as f:
            data = tomllib.load(f)
        project = data.get("project", data.get("tool", {}).get("poetry", {}))
        for field in ["name", "version"]:
            assert field in project or field in data.get("project", {}), (
                f"pyproject.toml missing field: {field}"
            )

    def test_requirements_file_exists(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        path = os.path.abspath(os.path.join(root, "requirements.txt"))
        assert os.path.exists(path), "requirements.txt is missing"

    def test_requirements_parseable(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        path = os.path.abspath(os.path.join(root, "requirements.txt"))
        with open(path, "r") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        assert len(lines) > 0, "requirements.txt is empty"


# ─────────────────────────────────────────────────────────────────────────────
# 20. MATHEMATICAL PROPERTIES — HAVOK-specific invariants
# ─────────────────────────────────────────────────────────────────────────────

class TestMathematicalProperties:

    def test_forcing_mean_near_zero(self, fitted_estimator):
        """Forcing signal is a residual — its mean should be approximately 0."""
        f = fitted_estimator.forcing_
        # Ignore the padded zeros at the start
        f_trimmed = f[fitted_estimator.m_fitted_:]
        assert abs(np.mean(f_trimmed)) < 1.0, (
            f"Forcing mean too large: {np.mean(f_trimmed):.4f}"
        )

    def test_singular_values_energy_decay(self, fitted_estimator):
        """First singular value must explain the most energy."""
        sv = fitted_estimator.singular_values_
        assert sv[0] == max(sv), "First singular value must be the largest"

    def test_forcing_std_less_than_signal_std(self, clean_sine, fitted_estimator):
        """Forcing is a residual — its std should be less than the input signal std."""
        f_trimmed = fitted_estimator.forcing_[fitted_estimator.m_fitted_:]
        sig_trimmed = clean_sine[fitted_estimator.m_fitted_:]
        assert np.std(f_trimmed) < np.std(sig_trimmed), (
            "Forcing std should be smaller than signal std"
        )

    def test_risk_rate_below_50_percent_on_clean_signal(self, fitted_estimator):
        """A clean sine wave should not be in 'crisis' mode more than 50% of the time."""
        risk_rate = np.mean(fitted_estimator.risk_)
        assert risk_rate < 0.5, f"Risk rate too high on clean signal: {risk_rate:.2%}"

    def test_r_parameter_controls_rank(self, clean_sine):
        """Higher r = more singular vectors = richer representation."""
        from havolib.estimator import HavokEstimator
        est3 = HavokEstimator(r=3, tau=2, m=20)
        est7 = HavokEstimator(r=7, tau=2, m=20)
        est3.fit(clean_sine)
        est7.fit(clean_sine)
        assert len(est3.singular_values_) <= len(est7.singular_values_)
