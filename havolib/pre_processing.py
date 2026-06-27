"""
Pre-processing pipeline for HAVOK (deeper layer requirement).

Includes:
- Interpolation for missing/NaN values
- Low-pass or Savitzky-Golay smoothing
- Outlier detection and removal (IQR or z-score)
- Detrending (optional)
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import pandas as pd
from typing import Tuple, Optional


def interpolate_missing(x: np.ndarray) -> np.ndarray:
    """Linear interpolation for NaNs / missing values."""
    x = np.asarray(x, dtype=float).copy()
    mask = np.isnan(x) | ~np.isfinite(x)
    if not np.any(mask):
        return x
    idx = np.arange(len(x))
    good = ~mask
    if np.sum(good) < 2:
        # fallback: fill with mean or zero
        x[mask] = np.nanmean(x) if np.any(good) else 0.0
        return x
    f = interp1d(idx[good], x[good], kind='linear', fill_value='extrapolate')
    x[mask] = f(idx[mask])
    return x


def remove_outliers(x: np.ndarray, method: str = 'iqr', threshold: float = 3.0) -> np.ndarray:
    """Remove/replace outliers. Returns cleaned series."""
    x = np.asarray(x, dtype=float).copy()
    if len(x) == 0:
        return x
    if method == 'iqr':
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (x < lower) | (x > upper)
    else:  # z-score
        z = np.abs((x - np.mean(x)) / (np.std(x) + 1e-12))
        mask = z > threshold
    
    if np.any(mask):
        # Replace with linear interpolation from neighbors
        x_clean = interpolate_missing(np.where(mask, np.nan, x))
        return x_clean
    return x


def smooth(x: np.ndarray, method: str = 'savgol', window: int = 11, poly: int = 3) -> np.ndarray:
    """Smoothing filter."""
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    if method == 'savgol':
        try:
            return signal.savgol_filter(x, window_length=window, polyorder=poly)
        except Exception:
            return x
    elif method == 'lowpass':
        # Simple butterworth lowpass (normalized freq 0.1)
        b, a = signal.butter(4, 0.1, btype='low', analog=False)
        return signal.filtfilt(b, a, x)
    return x


def preprocess(
    x: np.ndarray,
    interpolate: bool = True,
    smooth_method: Optional[str] = 'savgol',
    smooth_window: int = 11,
    outlier_method: Optional[str] = 'iqr',
    detrend: bool = False
) -> np.ndarray:
    """
    Full recommended pre-processing chain.
    Order: interpolate -> outliers -> smooth -> (optional) detrend
    """
    x = np.asarray(x, dtype=float).copy()
    
    if interpolate:
        x = interpolate_missing(x)
    
    if outlier_method:
        x = remove_outliers(x, method=outlier_method)
    
    if smooth_method:
        x = smooth(x, method=smooth_method, window=smooth_window)
    
    if detrend:
        x = signal.detrend(x)
    
    return x


if __name__ == "__main__":
    # demo
    np.random.seed(0)
    x = np.sin(np.linspace(0, 20, 500)) + np.random.normal(0, 0.3, 500)
    x[100:105] = np.nan
    x[300] = 10.0  # outlier
    x_clean = preprocess(x, smooth_method='savgol', outlier_method='iqr')
    print("Original has NaNs/outlier:", np.any(np.isnan(x)), np.max(np.abs(x)))
    print("Cleaned:", np.max(np.abs(x_clean)), "NaNs left:", np.sum(np.isnan(x_clean)))
    print("Pre-processing module ready.")
