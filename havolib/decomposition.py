from scipy.linalg import svd
import numpy as np

def eigen_time_delay(H: np.ndarray, r: int):
    """
    Perform truncated SVD on the Hankel matrix.

    Returns
    -------
    V : ndarray (n_rows, r)
        Top r eigen-time-delay coordinates (left singular vectors).
    s : ndarray (r,)
        Top singular values.
    """
    U, s, _ = svd(H, full_matrices=False)
    V = U[:, :r]
    return V, s[:r]
