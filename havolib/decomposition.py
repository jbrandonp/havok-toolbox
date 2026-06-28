"""Eigen-time-delay decomposition via SVD — GPU-accelerated when available."""
from typing import Optional
import warnings
from havolib.gpu import svd as _svd
import numpy as np


def eigen_time_delay(H: np.ndarray, r: int, solver: str = "auto", random_state: Optional[int] = None):
    """Perform truncated SVD on the Hankel matrix.

    Uses GPU (CuPy) if available (for 'auto'), otherwise NumPy/SciPy.
    Explicit 'scipy' forces CPU path for reproducibility.

    Parameters
    ----------
    H : ndarray
        Hankel matrix.
    r : int
        Number of components to keep (must be 0 < r <= min(H.shape)).
    solver : str
        'auto' (GPU if available else scipy), 'scipy' (force CPU), 'randomized'.
    random_state : int or None, optional
        Seed for the randomized solver (None for non-deterministic).

    Returns
    -------
    U : ndarray (n_rows, r) — Top r left singular vectors (eigen-time-delay coords)
    s : ndarray (r,) — Top singular values
    """
    hmin = min(H.shape)
    if not isinstance(r, int) or not (0 < r <= hmin):
        new_r = max(2, min(int(r) if isinstance(r, (int, np.integer)) else 2, hmin))
        warnings.warn(f"r={r} invalid for H shape {H.shape}; using r={new_r}")
        r = new_r

    if solver == "randomized":
        from sklearn.utils.extmath import randomized_svd
        U, s, Vt = randomized_svd(H, n_components=r, random_state=random_state)
        return U, s
    elif solver == "scipy":
        from scipy.linalg import svd as scipy_svd
        U, s, Vt = scipy_svd(H, full_matrices=False)
        return U[:, :r], s[:r]
    elif solver == "auto":
        U, s, Vt = _svd(H, full_matrices=False, r=r)
        return U, s
    else:
        raise ValueError(f"Unknown solver: {solver}. Supported: 'auto', 'scipy', 'randomized'")
