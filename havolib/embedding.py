import numpy as np

def hankel_matrix(data: np.ndarray, m: int, tau: int = 1) -> np.ndarray:
    """
    Construct a Hankel matrix for time-delay embedding.

    Parameters
    ----------
    data : 1D array
        The time series.
    m : int
        Number of columns (embedding dimension).
    tau : int
        Time delay in samples.

    Returns
    -------
    H : 2D array of shape (n - (m-1)*tau, m)
    """
    data = np.asarray(data).ravel()
    N = len(data)
    if N == 0:
        raise ValueError("Input data must not be empty.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Input data contains NaN or Inf values.")
    if m <= 0 or tau <= 0:
        raise ValueError("m and tau must be positive integers.")

    n_rows = N - (m - 1) * tau
    if n_rows <= 0:
        raise ValueError("Time series too short for given m and tau.")

    # Memory/resource guard (prevent OOM for huge m*tau)
    MAX_ELEMENTS = 50_000_000  # ~400 MB float64
    if n_rows * m > MAX_ELEMENTS:
        raise MemoryError(
            f"Requested Hankel matrix ({n_rows}x{m}) exceeds safe limit. "
            "Reduce m or tau (or downsample data)."
        )

    # Vectorized Hankel via indexing (works for any tau)
    # H[k, i] = data[ k + i * tau ]
    rows = np.arange(n_rows)[:, None]
    cols = np.arange(m) * tau
    idx = rows + cols
    H = data[idx]
    return H

def auto_tau(data: np.ndarray, max_lag: int = 100, method: str = "mi") -> int:
    """
    Optimal time delay tau.

    Parameters
    ----------
    data : 1D array
        The time series.
    max_lag : int
        Maximum lag to consider.
    method : {"mi", "autocorr"}
        "mi" uses first minimum of mutual information (Fraser & Swinney).
        "autocorr" uses first zero crossing of autocorrelation (faster but cruder).

    Returns
    -------
    int
        Optimal tau.
    """
    data = np.asarray(data).ravel()
    if data.size == 0:
        raise ValueError("Input data must not be empty.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Input data contains NaN or Inf values.")

    if method == "mi":
        try:
            from .auto_tune import optimal_tau_mi
        except ImportError:
            raise ImportError(
                "Method 'mi' requires the 'auto_tune' submodule. "
                "Install the full package or use method='autocorr'."
            )
        return optimal_tau_mi(data, max_lag=max_lag)
    elif method == "autocorr":
        # Original autocorrelation fallback (with safe normalization)
        N = len(data)
        lag_max = min(max_lag, N // 4)
        if lag_max < 1:
            return 1
        x = data - np.mean(data)
        autocorr = np.correlate(x, x, mode='full')[N-1 : N-1 + lag_max]
        autocorr = autocorr / (autocorr[0] + 1e-12)
        for lag in range(1, len(autocorr)):
            if autocorr[lag] <= 0:
                return lag
        return 1
    else:
        raise ValueError(f"Unknown method: '{method}'. Choose 'mi' or 'autocorr'.")
