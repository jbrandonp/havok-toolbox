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
    N = len(data)
    if N < m * tau:
        raise ValueError("Time series too short for given m and tau.")
    H = np.zeros((N - (m - 1) * tau, m))
    for i in range(m):
        start = i * tau
        H[:, i] = data[start : start + H.shape[0]]
    return H

def auto_tau(data: np.ndarray, max_lag: int = 100, method: str = "mi") -> int:
    """
    Optimal time delay tau.

    Parameters
    ----------
    method : "mi" (recommended) or "autocorr"
        "mi" uses first minimum of mutual information (Fraser & Swinney).
        "autocorr" uses first zero crossing of autocorrelation (faster but cruder).
    """
    if method == "mi":
        from .auto_tune import optimal_tau_mi
        return optimal_tau_mi(data, max_lag=max_lag)
    else:
        # Original autocorrelation fallback
        N = len(data)
        lag_max = min(max_lag, N // 4)
        x = data - np.mean(data)
        autocorr = np.correlate(x, x, mode='full')[N-1 : N-1 + lag_max]
        autocorr /= autocorr[0] if autocorr[0] != 0 else 1.0
        for lag in range(1, len(autocorr)):
            if autocorr[lag] <= 0:
                return lag
        return 1
