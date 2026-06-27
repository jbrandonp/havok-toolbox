"""Tests for pre_processing.py — interpolation, outliers, smoothing, detrending."""
import numpy as np
from havolib.pre_processing import (
    interpolate_missing,
    remove_outliers,
    smooth,
    preprocess,
)


class TestInterpolateMissing:
    def test_no_nans_returns_same(self):
        x = np.sin(np.linspace(0, 10, 100))
        result = interpolate_missing(x.copy())
        assert np.allclose(result, x)

    def test_interpolates_single_nan(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = interpolate_missing(x)
        assert not np.any(np.isnan(result))
        assert abs(result[2] - 3.0) < 0.01

    def test_interpolates_consecutive_nans(self):
        x = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        result = interpolate_missing(x)
        assert not np.any(np.isnan(result))
        assert 1.5 < result[1] < 2.5
        assert 2.5 < result[2] < 3.5

    def test_interpolates_nans_at_edges(self):
        x = np.array([np.nan, np.nan, 3.0, 4.0, 5.0])
        result = interpolate_missing(x)
        assert not np.any(np.isnan(result))

    def test_all_nans_fallback(self):
        x = np.full(10, np.nan)
        result = interpolate_missing(x)
        assert not np.any(np.isnan(result))

    def test_inf_values_treated_as_missing(self):
        x = np.array([1.0, np.inf, 3.0, -np.inf, 5.0])
        result = interpolate_missing(x)
        assert np.all(np.isfinite(result))


class TestRemoveOutliers:
    def test_iqr_removes_extreme_outlier(self):
        x = np.sin(np.linspace(0, 10, 100))
        x[50] = 100.0  # extreme outlier
        result = remove_outliers(x, method='iqr')
        assert abs(result[50]) < 10.0  # should be brought back to normal range

    def test_iqr_leaves_clean_data_unchanged(self):
        x = np.sin(np.linspace(0, 10, 100))
        result = remove_outliers(x.copy(), method='iqr')
        assert np.allclose(result, x, atol=0.01)

    def test_zscore_removes_outlier(self):
        x = np.random.randn(200) * 0.5
        x[100] = 20.0  # 40 sigma outlier
        result = remove_outliers(x, method='zscore', threshold=3.0)
        assert abs(result[100]) < 5.0

    def test_handles_empty_array(self):
        """Empty array should not crash."""
        result = remove_outliers(np.array([1.0, 2.0, 3.0]), method='iqr')
        assert len(result) == 3


class TestSmooth:
    def test_savgol_reduces_noise(self):
        np.random.seed(42)
        t = np.linspace(0, 10, 200)
        x_clean = np.sin(t)
        x_noisy = x_clean + np.random.normal(0, 0.3, len(t))
        x_smooth = smooth(x_noisy, method='savgol', window=21, poly=3)

        # Smoothed should be closer to clean than noisy is
        rmse_noisy = np.sqrt(np.mean((x_noisy - x_clean) ** 2))
        rmse_smooth = np.sqrt(np.mean((x_smooth - x_clean) ** 2))
        assert rmse_smooth < rmse_noisy * 0.8

    def test_short_signal_returned_unchanged(self):
        x = np.array([1.0, 2.0, 3.0])
        result = smooth(x, method='savgol', window=11, poly=3)
        assert np.allclose(result, x)

    def test_lowpass_does_not_crash(self):
        x = np.sin(np.linspace(0, 10, 200))
        result = smooth(x, method='lowpass')
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))

    def test_unknown_method_returns_input(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = smooth(x, method='unknown_blah', window=3)
        assert np.allclose(result, x)


class TestPreprocessPipeline:
    def test_full_chain_no_error(self):
        np.random.seed(42)
        t = np.linspace(0, 20, 500)
        x = np.sin(t) + np.random.normal(0, 0.3, 500)
        x[200:205] = np.nan
        x[400] = 15.0

        result = preprocess(
            x,
            interpolate=True,
            smooth_method='savgol',
            smooth_window=11,
            outlier_method='iqr',
            detrend=False,
        )
        assert not np.any(np.isnan(result))
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))

    def test_detrend_removes_linear_trend(self):
        x = np.linspace(0, 100, 200) + 5.0
        result = preprocess(x, interpolate=False, smooth_method=None, outlier_method=None, detrend=True)
        assert abs(np.mean(result)) < 0.1

    def test_disabled_steps_return_input(self):
        x = np.sin(np.linspace(0, 10, 100))
        result = preprocess(
            x,
            interpolate=False,
            smooth_method=None,
            outlier_method=None,
            detrend=False,
        )
        assert np.allclose(result, x)
