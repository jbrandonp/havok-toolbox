import numpy as np
import warnings
from numpy.linalg import lstsq

def extract_forcing(V: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Extract the intermittent forcing signal.

    Models the derivative of the last eigen-coordinate as a linear combination
    of the previous coordinates:

        dV_r / dt  ≈  Σ_{i=1}^{r-1} a_i * V_i  +  b   +   forcing(t)

    The residual after least-squares fit is returned as the forcing.

    Parameters
    ----------
    V : ndarray (n, r)
        Eigen-time-delay coordinates.
    t : ndarray (n,)
        Corresponding time vector (trimmed to match V). Must be strictly increasing.

    Returns
    -------
    forcing : ndarray (n,)
    """
    n, r = V.shape
    if r < 2:
        raise ValueError("r must be at least 2 to separate forcing.")

    # Input validation (Critical/Major robustness)
    if len(t) != n:
        raise ValueError(f"Time vector length ({len(t)}) must match V rows ({n}).")
    if not np.all(np.diff(t) > 0):
        raise ValueError("Time vector must be strictly increasing (monotonic).")
    if np.any(~np.isfinite(V)) or np.any(~np.isfinite(t)):
        raise ValueError("Input arrays contain NaN or Inf values.")

    # Safety size limits (prevent DoS / OOM in production)
    MAX_N = 1_000_000
    MAX_R = 500
    if n > MAX_N or r > MAX_R:
        raise ValueError(f"Input size exceeds safety limits (n≤{MAX_N}, r≤{MAX_R}). "
                         "Reduce embedding size or downsample.")

    # Underdetermined warning (Minor robustness)
    if n < r:
        warnings.warn("Underdetermined system (n < r); forcing may be unstable.", RuntimeWarning)

    # Optimised: compute gradient ONLY for the target (last) coordinate (Major perf)
    y = np.gradient(V[:, -1], t)

    # Predictors: all but last coordinate + bias term
    # Slightly more memory-efficient construction
    X = np.empty((n, r), dtype=float)
    X[:, :-1] = V[:, :-1]
    X[:, -1] = 1.0

    coeffs, _, _, _ = lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    forcing = y - y_pred
    return forcing
