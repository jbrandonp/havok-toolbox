"""
Baseline methods for regime-shift detection benchmark.

Includes: rolling std, CUSUM, ARIMA residual, and simple threshold.
Each baseline returns a risk-like signal (0-1 or anomaly score).
"""

import numpy as np
from typing import Tuple


def rolling_std_detector(
    x: np.ndarray,
    window: int = 100,
    n_std: float = 3.0,
) -> np.ndarray:
    """Detect shifts as points where rolling std exceeds n_std * global std."""
    if len(x) < window:
        return np.zeros(len(x))

    global_std = np.std(x)
    if global_std < 1e-12:
        return np.zeros(len(x))

    risk = np.zeros(len(x))
    for i in range(window, len(x)):
        local_std = np.std(x[i - window:i])
        if local_std > n_std * np.std(x[:window]):
            risk[i] = min(1.0, local_std / (n_std * np.std(x[:window]) + 1e-12))

    return risk


def cusum_detector(
    x: np.ndarray,
    drift: float = 1.0,
    threshold: float = 5.0,
) -> np.ndarray:
    """CUSUM (Cumulative Sum) change-point detector."""
    if len(x) < 10:
        return np.zeros(len(x))

    mu0 = np.mean(x[:len(x) // 4])
    risk = np.zeros(len(x))
    s_high, s_low = 0.0, 0.0

    for i in range(len(x)):
        s_high = max(0, s_high + x[i] - mu0 - drift)
        s_low = max(0, s_low - x[i] + mu0 - drift)
        risk[i] = min(1.0, max(s_high, s_low) / threshold)

    return risk


def arima_residual_detector(
    x: np.ndarray,
    window: int = 200,
    n_std: float = 3.0,
) -> np.ndarray:
    """Simple AR(1) residual-based detector."""
    if len(x) < window + 2:
        return np.zeros(len(x))

    risk = np.zeros(len(x))
    for i in range(window, len(x)):
        train = x[i - window:i]
        # AR(1): x_t = mu + phi * x_{t-1} + eps
        phi = np.corrcoef(train[:-1], train[1:])[0, 1]
        if abs(phi) > 0.99:
            phi = 0.5
        mu = np.mean(train)
        pred = mu + phi * (x[i - 1] - mu)
        residual = abs(x[i] - pred)
        train_resid = np.abs(train[1:] - (mu + phi * (train[:-1] - mu)))
        resid_std = np.std(train_resid)
        if resid_std > 1e-12:
            risk[i] = min(1.0, residual / (n_std * resid_std))

    return risk


def simple_threshold_detector(
    x: np.ndarray,
    window: int = 200,
    n_std: float = 3.0,
) -> np.ndarray:
    """Simple absolute-value threshold detector."""
    if len(x) < window:
        return np.zeros(len(x))

    baseline = np.std(x[:window])
    if baseline < 1e-12:
        return np.zeros(len(x))

    risk = np.zeros(len(x))
    for i in range(window, len(x)):
        risk[i] = min(1.0, abs(x[i]) / (n_std * baseline))

    return risk


BASELINES = {
    "rolling_std": (rolling_std_detector, "Rolling standard deviation"),
    "cusum": (cusum_detector, "CUSUM change-point"),
    "arima_residual": (arima_residual_detector, "AR(1) residual"),
    "simple_threshold": (simple_threshold_detector, "Simple amplitude threshold"),
}
