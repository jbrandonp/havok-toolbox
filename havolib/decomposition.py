"""Eigen-time-delay decomposition via SVD — GPU-accelerated when available."""
from havolib.gpu import svd as _svd
import numpy as np


def eigen_time_delay(H: np.ndarray, r: int):
    """Perform truncated SVD on the Hankel matrix.

    Uses GPU (CuPy) if available, otherwise NumPy/SciPy.

    Returns
    -------
    V : ndarray (n_rows, r)
        Top r eigen-time-delay coordinates (left singular vectors).
    s : ndarray (r,)
        Top singular values.
    """
    U, s, Vt = _svd(H, full_matrices=False, r=r)
    return U, s
