"""
Auto-tuning for HAVOK embedding parameters.

- Mutual Information for optimal tau (proper method)
- Improved FNN (False Nearest Neighbors) for m selection
"""

import numpy as np
from typing import Tuple
from scipy.spatial import cKDTree


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 32) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    joint, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    joint = joint.T
    px = np.sum(joint, axis=0)
    py = np.sum(joint, axis=1)
    joint = np.where(joint > 0, joint, 1e-12)
    px = np.where(px > 0, px, 1e-12)
    py = np.where(py > 0, py, 1e-12)
    return float(np.sum(joint * (np.log(joint) - np.log(px) - np.log(py[:, None]))))


def optimal_tau_mi(data: np.ndarray, max_lag: int = 100, bins: int = 32) -> int:
    data = np.asarray(data).ravel()
    data = (data - np.mean(data)) / (np.std(data) + 1e-12)
    N = len(data)
    max_lag = min(max_lag, N // 4)
    mis = np.array([mutual_information(data[:-lag], data[lag:], bins=bins) for lag in range(1, max_lag + 1)])
    for i in range(1, len(mis) - 1):
        if mis[i] < mis[i-1] and mis[i] < mis[i+1]:
            return i + 1
    threshold = mis[0] / np.e
    for i, v in enumerate(mis):
        if v < threshold:
            return i + 1
    return 1


def false_nearest_neighbors(data: np.ndarray, tau: int, max_m: int = 50, rtol: float = 15.0, atol: float = 2.0) -> np.ndarray:
    """
    Basic False Nearest Neighbors (Kennel et al.).
    Returns array of FNN fraction for each m from 1 to max_m.
    """
    data = np.asarray(data).ravel()
    N = len(data)
    fnn_frac = []

    for m in range(1, max_m + 1):
        # Build delay vectors
        M = N - (m + 1) * tau
        if M < 100:
            break

        # Create matrix of delay vectors
        X = np.array([data[i:i + m * tau:tau] for i in range(M)])
        X_next = data[(m * tau):(m * tau + M)]

        # Find nearest neighbor in m dimensions
        tree = cKDTree(X)
        dist, idx = tree.query(X, k=2)
        nn_dist = dist[:, 1]
        nn_idx = idx[:, 1]

        # Check if they are false neighbors in m+1
        d_next = np.abs(X_next - X_next[nn_idx])
        fnn = (d_next > rtol * nn_dist) | (d_next / (np.std(data) + 1e-12) > atol)

        frac = np.mean(fnn)
        fnn_frac.append(frac)

        if frac < 0.05:   # common threshold
            break

    return np.array(fnn_frac)


def optimal_m_fnn(data: np.ndarray, tau: int, max_m: int = 50) -> int:
    """Return the smallest m where FNN fraction drops low."""
    fracs = false_nearest_neighbors(data, tau, max_m=max_m)
    for i, f in enumerate(fracs):
        if f < 0.05:
            return i + 1
    # fallback
    return min(max(5, len(data) // 15), max_m)


def suggest_parameters(data: np.ndarray, max_lag: int = 100, max_m: int = 50) -> dict:
    """
    Returns good tau (using Mutual Information) and m using improved FNN.
    """
    data = np.asarray(data).ravel()
    data = data - np.mean(data)

    tau = optimal_tau_mi(data, max_lag=max_lag)
    m = optimal_m_fnn(data, tau, max_m=max_m)

    return {
        "tau": int(tau),
        "m": int(m),
        "method": "mutual_information + FNN",
        "recommendation": f"tau={tau}, m={m} (MI for tau, FNN for m)"
    }


if __name__ == "__main__":
    from data_loader import generate_lorenz
    _, x = generate_lorenz(n_points=1500)
    print(suggest_parameters(x))
