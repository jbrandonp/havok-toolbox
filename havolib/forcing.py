import numpy as np
from numpy.linalg import lstsq

def extract_forcing(V: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Extract the intermittent forcing signal.

    Models the derivative of the last eigen-coordinate as a linear combination
    of the previous coordinates. The residual is the forcing.

    Parameters
    ----------
    V : ndarray (n, r)
        Eigen-time-delay coordinates.
    t : ndarray (n,)
        Corresponding time vector (trimmed to match V).

    Returns
    -------
    forcing : ndarray (n,)
    """
    n, r = V.shape
    if r < 2:
        raise ValueError("r must be at least 2 to separate forcing.")

    dv = np.gradient(V, t, axis=0)

    # Predictors: all but last coordinate + bias term
    X = np.column_stack([V[:, :-1], np.ones(n)])
    y = dv[:, -1]

    coeffs, _, _, _ = lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    forcing = y - y_pred
    return forcing
