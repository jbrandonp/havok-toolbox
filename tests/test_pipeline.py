"""Tests for the full HAVOK pipeline integration."""
import numpy as np
from havolib.pipeline import HavokPipeline
from havolib.data_loader import generate_lorenz, generate_eeg_like


class TestHavokPipelineIntegration:
    def test_full_pipeline_lorenz(self):
        """End-to-end: Lorenz → fit → forcing → risk."""
        t, x = generate_lorenz(n_points=3000)

        pipeline = HavokPipeline(tau=1, m=50, r=5, threshold_std=3.0, window=100)
        pipeline.fit(t, x)

        forcing = pipeline.get_forcing()
        risk = pipeline.get_risk()
        V = pipeline.get_eigen_coordinates()

        assert len(forcing) > 0
        assert len(risk) == len(forcing)
        # r may be adjusted if auto-tune selects m < r
        assert V.shape[1] >= 2, f"Expected at least 2 eigen-coordinates, got {V.shape[1]}"
        assert np.max(np.abs(forcing)) > 0.005, f"Forcing too low: {np.max(np.abs(forcing)):.6f}"

    def test_full_pipeline_with_preprocessing(self):
        """Pre-processing should not crash on clean data."""
        t, x = generate_lorenz(n_points=2000)

        pipeline = HavokPipeline(
            do_preprocess=True,
            interpolate=True,
            smooth_method='savgol',
            smooth_window=11,
            outlier_method='iqr',
        )
        pipeline.auto_fit(t, x)

        forcing = pipeline.get_forcing()
        assert len(forcing) > 0
        assert np.all(np.isfinite(forcing))

    def test_surrogate_validation_integration(self):
        """Surrogate validation should run on Lorenz and return significant result."""
        t, x = generate_lorenz(n_points=1500)

        pipeline = HavokPipeline(tau=10, m=30, r=5)
        pipeline.fit(t, x)

        summary = pipeline.validate_with_surrogates(n_surrogates=10)
        assert "observed_max_forcing" in summary
        assert "p_value" in summary
        assert "significant_at_alpha" in summary
        assert summary["n_surrogates"] == 10

    def test_get_forcing_before_fit_raises(self):
        pipeline = HavokPipeline()
        try:
            pipeline.get_forcing()
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_get_risk_before_fit_raises(self):
        pipeline = HavokPipeline()
        try:
            pipeline.get_risk()
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_suggest_parameters_lorenz(self):
        _, x = generate_lorenz(n_points=1000)
        pipeline = HavokPipeline()
        params = pipeline.suggest_parameters(x, max_lag=50, max_m=30)
        assert params["tau"] > 0
        assert params["m"] >= 2

    def test_manual_tau_m_override(self):
        t, x = generate_lorenz(n_points=2000)
        pipeline = HavokPipeline(tau=7, m=25, r=4)
        pipeline.fit(t, x)
        assert pipeline.tau == 7
        assert pipeline.m == 25

    def test_data_length_warning(self):
        """Very short data should still work but warn."""
        t = np.arange(100)
        x = np.sin(t * 0.1)
        pipeline = HavokPipeline(tau=1, m=50, r=3)
        pipeline.fit(t, x)  # should not crash despite length warning

    def test_reproducibility(self):
        """Same data + same params → same forcing."""
        t, x = generate_lorenz(n_points=2000)
        p1 = HavokPipeline(tau=10, m=30, r=5)
        p1.fit(t, x)
        p2 = HavokPipeline(tau=10, m=30, r=5)
        p2.fit(t, x)
        assert np.allclose(p1.get_forcing(), p2.get_forcing())
