import numpy as np
import warnings
from typing import List

def _rolling_std(arr: np.ndarray, window: int, min_periods: int, ddof: int = 0) -> np.ndarray:
    """Pure NumPy rolling population std (ddof=0) with min_periods support. O(N) time."""
    n = len(arr)
    result = np.full(n, np.nan, dtype=float)
    if n == 0 or window < 1 or min_periods < 1:
        return result
    if min_periods > window:
        min_periods = window
    # Vectorized using stride tricks for windows (pure numpy, no pandas)
    if window > n:
        return result
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        wins = sliding_window_view(arr, window)
        # wins shape (n-window+1, window)
        # Assign to positions from window-1 onward
        wsum = wins.sum(axis=1)
        wsum2 = (wins * wins).sum(axis=1)
        wcnt = window
        mean = wsum / wcnt
        var = (wsum2 / wcnt) - mean * mean
        stds = np.sqrt(np.maximum(var, 0.0))
        # leading min_periods-1 remain nan, then full window stds
        start_assign = window - 1
        result[start_assign : start_assign + len(stds)] = stds
        # For positions with partial windows (min_periods <= cnt < window), use cumsum prefix for early
        if min_periods < window:
            cum = np.cumsum(arr, dtype=float)
            cum2 = np.cumsum(arr * arr, dtype=float)
            for i in range(min_periods-1, min(start_assign, n)):
                start = 0
                cnt = i + 1
                if cnt >= min_periods:
                    s = cum[i]
                    s2 = cum2[i]
                    mn = s / cnt
                    vr = (s2 / cnt) - mn*mn
                    result[i] = np.sqrt(max(vr, 0.0))
    except Exception:
        # Fallback to O(n) cumsum + python (still fast enough, no 3rd party)
        cumsum = np.cumsum(arr, dtype=float)
        cumsum2 = np.cumsum(arr * arr, dtype=float)
        for i in range(min_periods - 1, n):
            start = max(0, i - window + 1)
            cnt = i - start + 1
            if cnt >= min_periods:
                s = cumsum[i] - (cumsum[start - 1] if start > 0 else 0)
                s2 = cumsum2[i] - (cumsum2[start - 1] if start > 0 else 0)
                mean = s / cnt
                var = (s2 / cnt) - mean * mean
                result[i] = np.sqrt(max(var, 0.0))
    return result

def threshold_risk(forcing: np.ndarray,
                   window: int = 100,
                   n_std: float = 3.0) -> np.ndarray:
    """
    Binary risk flag: 1 when |forcing| exceeds rolling n_std * std.

    Uses pure NumPy rolling std (population, ddof=0) for efficiency.
    The output has the same length as `forcing`. Positions where the
    rolling std is undefined (leading samples before min_periods) are set to 0.

    Raises
    ------
    ValueError
        If `forcing` contains NaN/Inf, if window < 2, n_std <= 0,
        or if the input exceeds internal size limits.
    """
    forcing = np.asarray(forcing, dtype=float)
    if forcing.ndim != 1:
        raise ValueError("forcing must be a 1-D array")
    if not np.isfinite(forcing).all():
        raise ValueError("forcing contains NaN or Inf values")
    if n_std <= 0:
        raise ValueError("n_std must be positive")
    if window < 2:
        raise ValueError("window must be at least 2")

    MAX_SAMPLES = 10_000_000
    if forcing.size > MAX_SAMPLES:
        raise ValueError(f"Input exceeds maximum allowed size ({MAX_SAMPLES})")

    abs_f = np.abs(forcing)
    min_periods = max(5, window // 5)
    if min_periods > window:
        raise ValueError(f"min_periods ({min_periods}) cannot exceed window ({window})")

    rolling_std = _rolling_std(abs_f, window, min_periods, ddof=0)
    threshold = n_std * rolling_std
    risk = (abs_f > threshold).astype(int)
    # Mask only positions where rolling std is undefined (NaN)
    risk[np.isnan(rolling_std)] = 0
    return risk

def pelt_changepoint(forcing: np.ndarray, penalty: float = 10.0) -> List[int]:
    """
    Offline change-point detection on the forcing signal using ruptures.Pelt
    with RBF cost model.  PELT (Pruned Exact Linear Time) is a frequentist
    penalised-cost method, NOT a Bayesian changepoint detector.

    Unlike BOCPD (Bayesian Online Changepoint Detection), PELT works on the
    full signal at once and selects change-points by minimising a cost
    function with a linear penalty term.  It was chosen here because it is
    deterministic, fast (O(n) expected), and produces reproducible results
    without requiring prior distributions on run-lengths.

    The final artificial change-point equal to len(forcing) returned by
    ruptures is stripped so only internal break indices are returned.

    Raises
    ------
    ValueError
        If penalty <= 0, forcing contains non-finite values, or input exceeds size limit.
    """
    forcing = np.asarray(forcing, dtype=float)
    if forcing.ndim != 1:
        raise ValueError("forcing must be a 1-D array")
    if not np.isfinite(forcing).all():
        raise ValueError("forcing contains NaN or Inf values")
    if penalty <= 0:
        raise ValueError("penalty must be positive")

    MAX_SAMPLES = 10_000_000
    if forcing.size > MAX_SAMPLES:
        raise ValueError(f"Input exceeds maximum allowed size ({MAX_SAMPLES})")

    try:
        import ruptures as rpt
    except ImportError:
        raise ImportError("ruptures package required for pelt_changepoint. "
                          "pip install ruptures")

    algo = rpt.Pelt(model="rbf", min_size=10, jump=5).fit(forcing.reshape(-1, 1))
    cps = algo.predict(pen=penalty)
    # Strip the artificial terminal point (len(forcing))
    return [cp for cp in cps if cp < len(forcing)]


# Backward-compatible alias (deprecated — use pelt_changepoint)
def bayesian_changepoint(forcing: np.ndarray, penalty: float = 10.0) -> List[int]:
    """Deprecated: use pelt_changepoint() instead."""
    import warnings
    warnings.warn("bayesian_changepoint is deprecated, use pelt_changepoint", DeprecationWarning, stacklevel=2)
    return pelt_changepoint(forcing, penalty=penalty)
