"""Tests for detection.py — threshold_risk and bayesian_changepoint."""
import numpy as np
from havolib.detection import threshold_risk, bayesian_changepoint


class TestThresholdRisk:
    def test_no_risk_when_forcing_is_zero(self):
        forcing = np.zeros(200)
        risk = threshold_risk(forcing, window=50, n_std=3.0)
        assert np.sum(risk) == 0

    def test_clear_spike_flagged(self):
        forcing = np.zeros(200)
        forcing[100] = 10.0  # clear spike
        risk = threshold_risk(forcing, window=50, n_std=2.0)
        assert risk[100] == 1, "Spike should be flagged as risk"

    def test_output_is_binary(self):
        forcing = np.random.randn(500)
        risk = threshold_risk(forcing, window=100, n_std=3.0)
        assert np.all((risk == 0) | (risk == 1))

    def test_first_window_values_are_zero(self):
        forcing = np.random.randn(300)
        risk = threshold_risk(forcing, window=50, n_std=3.0)
        assert np.all(risk[:50] == 0)

    def test_higher_threshold_fewer_events(self):
        forcing = np.random.randn(1000)
        forcing[500] = 8.0  # one strong spike
        risk_low = threshold_risk(forcing, window=100, n_std=1.0)
        risk_high = threshold_risk(forcing, window=100, n_std=5.0)
        assert np.sum(risk_low) >= np.sum(risk_high)

    def test_handles_nans_in_forcing(self):
        forcing = np.array([0.0, 1.0, np.nan, 1.0, 0.0])
        risk = threshold_risk(forcing, window=5, n_std=3.0)
        assert len(risk) == 5


class TestBayesianChangepoint:
    def test_returns_list_of_int(self):
        forcing = np.random.randn(200)
        forcing[100:120] += 2.0  # artificial shift
        cps = bayesian_changepoint(forcing)
        assert isinstance(cps, list)
        assert len(cps) > 0
        assert all(isinstance(c, (int, np.integer)) for c in cps)

    def test_detects_injected_changepoint(self):
        forcing = np.zeros(300)
        forcing[150:] = 5.0  # clear change at 150
        cps = bayesian_changepoint(forcing, penalty=5.0)
        # Should detect a change near 150
        any_near_150 = any(140 < c <= 160 for c in cps)
        assert any_near_150, f"No changepoint near 150 in {cps}"

    def test_final_index_is_last_point(self):
        forcing = np.random.randn(100)
        cps = bayesian_changepoint(forcing)
        assert cps[-1] == len(forcing) or cps[-1] >= len(forcing) - 5
