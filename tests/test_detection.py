"""Tests for detection.py — threshold_risk and bayesian_changepoint."""
import numpy as np
from havolib.detection import threshold_risk, pelt_changepoint


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

    def test_leading_undefined_risk_is_zero(self):
        forcing = np.random.randn(300)
        risk = threshold_risk(forcing, window=50, n_std=3.0)
        # Only the leading positions where rolling std is NaN (before min_periods) are forced to 0.
        min_periods = max(5, 50 // 5)
        # The first min_periods-1 are guaranteed nan in rolling
        assert np.all(risk[:min_periods-1] == 0)
        # Value at min_periods-1 may be 0 or 1 depending on data
        assert len(risk) == 300

    def test_higher_threshold_fewer_events(self):
        forcing = np.random.randn(1000)
        forcing[500] = 8.0  # one strong spike
        risk_low = threshold_risk(forcing, window=100, n_std=1.0)
        risk_high = threshold_risk(forcing, window=100, n_std=5.0)
        assert np.sum(risk_low) >= np.sum(risk_high)

    def test_handles_nans_in_forcing(self):
        forcing = np.array([0.0, 1.0, np.nan, 1.0, 0.0])
        # Per audit: NaN/Inf now raises (prevents silent bad results)
        try:
            threshold_risk(forcing, window=5, n_std=3.0)
            assert False, "Should raise on NaN"
        except ValueError:
            pass


class TestChangepoint:
    def test_returns_list_of_int(self):
        forcing = np.random.randn(200)
        forcing[120:140] += 10.0
        cps = pelt_changepoint(forcing, penalty=5.0)
        assert isinstance(cps, list)
        assert len(cps) > 0
        assert all(isinstance(c, (int, np.integer)) for c in cps)
        assert len(forcing) not in cps

    def test_detects_injected_changepoint(self):
        forcing = np.zeros(300)
        forcing[150:] = 5.0
        cps = pelt_changepoint(forcing, penalty=5.0)
        any_near_150 = any(140 < c <= 160 for c in cps)
        assert any_near_150, f"No changepoint near 150 in {cps}"

    def test_no_artificial_endpoint(self):
        forcing = np.random.randn(100)
        cps = pelt_changepoint(forcing)
        assert len(forcing) not in cps
        assert all(0 <= c < len(forcing) for c in cps)
