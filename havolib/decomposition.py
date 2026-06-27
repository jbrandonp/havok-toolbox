"""Eigen-time-delay decomposition via SVD — GPU-accelerated when available."""
from havolib.gpu import svd as _svd
import numpy as np


def eigen_time_delay(H: np.ndarray, r: int, solver: str = "auto"):
    """Perform truncated SVD on the Hankel matrix.

    Uses GPU (CuPy) if available, otherwise NumPy/SciPy.

    Parameters
    ----------
    solver : str — 'auto' (GPU if avail), 'scipy', 'randomized'

    Returns
    -------
    V : ndarray (n_rows, r) — Top r left singular vectors
    s : ndarray (r,) — Top singular values
    """
    if solver == "randomized":
        from sklearn.utils.extmath import randomized_svd
        U, s, Vt = randomized_svd(H, n_components=r, random_state=42)
        return U, s
    else:
        U, s, Vt = _svd(H, full_matrices=False, r=r)
        return U, s
