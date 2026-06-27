"""Tests for config module — dataclass validation + YAML profiles."""
import pytest
import tempfile, os
from havolib.config import (
    HavokParams, PreprocessingConfig, PipelineConfig, StreamConfig,
    EngineConfig, AlertTargetConfig, load_profile, list_profiles,
)

class TestHavokParams:
    def test_valid_defaults(self):
        hp = HavokParams()
        assert hp.tau == 1 and hp.m == 50 and hp.r == 5

    def test_rejects_tau_zero(self):
        with pytest.raises(ValueError): HavokParams(tau=0)

    def test_rejects_m_too_small(self):
        with pytest.raises(ValueError): HavokParams(m=1)

    def test_rejects_r_zero(self):
        with pytest.raises(ValueError): HavokParams(r=0)

    def test_rejects_r_exceeds_m(self):
        with pytest.raises(ValueError): HavokParams(m=5, r=10)

    def test_rejects_negative_threshold(self):
        with pytest.raises(ValueError): HavokParams(threshold_std=-1)

    def test_rejects_small_window(self):
        with pytest.raises(ValueError): HavokParams(window=5)

    def test_to_dict_roundtrip(self):
        hp = HavokParams(tau=10, m=30, r=5, threshold_std=3.5, window=150)
        d = hp.to_dict()
        assert d["tau"] == 10 and d["m"] == 30

class TestPreprocessingConfig:
    def test_valid(self):
        pp = PreprocessingConfig()
        assert pp.smooth_method == "savgol"

    def test_rejects_bad_smooth_method(self):
        with pytest.raises(ValueError): PreprocessingConfig(smooth_method="invalid")

    def test_rejects_even_window(self):
        with pytest.raises(ValueError): PreprocessingConfig(smooth_window=10)

    def test_rejects_bad_outlier(self):
        with pytest.raises(ValueError): PreprocessingConfig(outlier_method="bad")

class TestPipelineConfig:
    def test_default(self):
        pc = PipelineConfig()
        assert pc.auto_tune is True

    def test_from_dict_roundtrip(self):
        d = {"havok": {"tau": 5, "m": 40, "r": 6}, "preprocessing": {"smooth_method": None},
             "auto_tune": False}
        pc = PipelineConfig.from_dict(d)
        assert pc.havok.tau == 5
        assert not pc.auto_tune

class TestEngineConfig:
    def test_from_dict(self):
        d = {"streams": [{"id": "s1", "source": "csv://data.csv"}],
             "alert_targets": {"console": {"type": "stdout"}}}
        ec = EngineConfig.from_dict(d)
        assert len(ec.streams) == 1
        assert "console" in ec.alert_targets

    def test_rejects_bad_log_level(self):
        with pytest.raises(ValueError):
            EngineConfig.from_dict({"log_level": "TRACE", "streams": []})

class TestLoadProfile:
    def test_loads_builtin_profiles(self):
        profiles = list_profiles()
        assert "eeg" in profiles
        assert "finance" in profiles

    def test_eeg_profile_params(self):
        cfg = load_profile("eeg")
        assert cfg.havok.tau == 5

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            load_profile("nonexistent_profile_xyz")
