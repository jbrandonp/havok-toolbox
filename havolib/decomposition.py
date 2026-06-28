"""Eigen-time-delay decomposition via SVD — GPU-accelerated when available.

Paper correspondence: This module implements Step 2 of Brunton et al. 2017 —
truncated SVD of the Hankel matrix H ≈ U Σ V^T.  The returned U[:, :r]
(left singular vectors) are the temporal eigen-time-delay coordinates
called V(t) in the paper.  The naming difference arises from numpy's
SVD convention (H = U Σ V^T) vs the paper's notation where V(t) denotes
the temporal coordinates extracted from the Hankel matrix.
"""
from typing import Optional, Tuple
import warnings
from havolib.gpu import svd as _svd
import numpy as np


def eigen_time_delay(
    H: np.ndarray, r: int, solver: str = "auto", random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Truncated SVD of Hankel matrix → eigen-time-delay coordinates.

    Computes H ≈ U Σ V^T and returns U[:, :r] and Σ[:r].  In the standard
    numpy convention U carries time in its rows, so the returned array is
    the temporal coordinate matrix the paper calls V(t).

    Paper (Brunton et al. 2017, §Methods):
        "The eigen-time-delay coordinates are obtained by taking the SVD
         of the Hankel matrix H = U Σ V^T.  We use the columns of U..."


    Parameters
    ----------
    H : ndarray, shape (n, m)
        Hankel matrix from the delay-embedding step.
    r : int
        Number of dominant singular modes to retain (1 < r ≤ min(n, m)).
    solver : str
        'auto'       → GPU (CuPy) if available, else SciPy exact SVD
        'scipy'      → force CPU via scipy.linalg.svd
        'randomized' → sklearn.randomized_svd (approximate, fast on large H)
    random_state : int or None
        Seed for the randomized solver.  Ignored otherwise.

    Returns
    -------
    U_r : ndarray, shape (n, r)
        Left singular vectors = eigen-time-delay coordinates.
        In the paper these are called V(t) — the temporal coordinates
        used for the linear forcing model.
    s_r : ndarray, shape (r,)
        Top r singular values, sorted descending.
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
        raise ValueError(
            f"Unknown solver: {solver!r}. Supported: 'auto', 'scipy', 'randomized'"
        )
