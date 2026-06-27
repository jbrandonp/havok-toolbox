"""Tests for ml_risk_predictor.py — ESN-based forcing risk prediction."""
import numpy as np
from havolib.ml_risk_predictor import ForcingRiskPredictor, quick_forcing_risk


class TestForcingRiskPredictor:
    def test_initialization(self):
        model = ForcingRiskPredictor(reservoir_size=100, spectral_radius=0.9, leak_factor=0.3)
        assert model.N == 100
        assert model.rho == 0.9
        assert model.alpha == 0.3
        assert not model._fitted

    def test_train_and_predict(self):
        t = np.linspace(0, 20 * np.pi, 1000)
        forcing = np.sin(t) + 0.3 * np.random.randn(1000)

        model = ForcingRiskPredictor(reservoir_size=100, random_seed=42)
        model.train(forcing)

        assert model._fitted

        preds, risk = model.predict(forcing[-200:], horizon=30)
        assert len(preds) == 30
        assert 0.0 <= risk <= 1.0
        assert np.all(np.isfinite(preds))

    def test_predict_before_train_raises(self):
        model = ForcingRiskPredictor()
        try:
            model.predict(np.random.randn(100), horizon=10)
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_different_seeds_produce_different_models(self):
        t = np.linspace(0, 10 * np.pi, 500)
        forcing = np.sin(t) + 0.2 * np.random.randn(500)

        m1 = ForcingRiskPredictor(reservoir_size=80, random_seed=1)
        m1.train(forcing)
        p1, _ = m1.predict(forcing[-100:], horizon=15)

        m2 = ForcingRiskPredictor(reservoir_size=80, random_seed=2)
        m2.train(forcing)
        p2, _ = m2.predict(forcing[-100:], horizon=15)

        # Different seeds should produce different predictions
        assert not np.allclose(p1, p2, atol=1e-6)

    def test_reproducibility_same_seed(self):
        t = np.linspace(0, 10 * np.pi, 500)
        forcing = np.sin(t) + 0.2 * np.random.randn(500)

        m1 = ForcingRiskPredictor(reservoir_size=80, random_seed=42)
        m1.train(forcing)
        p1, _ = m1.predict(forcing[-100:], horizon=15)

        m2 = ForcingRiskPredictor(reservoir_size=80, random_seed=42)
        m2.train(forcing)
        p2, _ = m2.predict(forcing[-100:], horizon=15)

        assert np.allclose(p1, p2)

    def test_low_risk_for_constant_signal(self):
        forcing = np.zeros(500) + 0.01 * np.random.randn(500)
        model = ForcingRiskPredictor(reservoir_size=80, random_seed=42)
        model.train(forcing)
        _, risk = model.predict(forcing[-100:], horizon=30)
        assert risk < 0.5, f"Expected low risk for zero signal, got {risk:.3f}"

    def test_high_risk_for_spiky_signal(self):
        t = np.linspace(0, 20 * np.pi, 1000)
        forcing = np.sin(t)
        forcing[400:450] = 5.0  # big spike
        forcing[700:750] = -4.0  # another spike

        model = ForcingRiskPredictor(reservoir_size=100, random_seed=42)
        model.train(forcing[:600])  # train on part with first spike
        _, risk = model.predict(forcing[500:700], horizon=30)
        assert risk > 0.0  # should detect some risk


class TestQuickForcingRisk:
    def test_returns_expected_keys(self):
        t = np.linspace(0, 10 * np.pi, 800)
        forcing = np.sin(t) + 0.2 * np.random.randn(800)
        forcing[600:620] += 3.0

        result = quick_forcing_risk(forcing, horizon=20, reservoir_size=80)
        assert "predicted_forcing" in result
        assert "regime_shift_risk" in result
        assert "horizon" in result
        assert result["horizon"] == 20
        assert len(result["predicted_forcing"]) == 20
        assert 0.0 <= result["regime_shift_risk"] <= 1.0

    def test_works_with_small_reservoir(self):
        forcing = np.random.randn(300)
        result = quick_forcing_risk(forcing, horizon=10, reservoir_size=30)
        assert result["predicted_forcing"] is not None
