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
    if len(x) == 0 or len(y) == 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Inputs to mutual_information must be non-empty and finite.")
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
    if len(data) == 0 or not np.all(np.isfinite(data)):
        raise ValueError("Data for optimal_tau_mi must be non-empty and finite.")
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
    False Nearest Neighbors (Kennel et al.).
    Returns array of FNN fraction for each m from 1 to max_m.

    NOTE: FNN finds the MINIMAL embedding dimension to unfold the attractor.
    HAVOK needs MORE dimensions (≥15) for a good linear Koopman approximation.
    Use optimal_m_havok() for HAVOK parameter selection, not optimal_m_fnn().
    """
    data = np.asarray(data).ravel()
    if len(data) == 0 or not np.all(np.isfinite(data)):
        raise ValueError("Data for FNN must be non-empty and finite.")
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

        if frac < 0.01:   # stricter threshold — continue until truly stable
            if len(fnn_frac) >= 3 and all(f < 0.02 for f in fnn_frac[-3:]):
                break

    return np.array(fnn_frac)


def optimal_m_fnn(data: np.ndarray, tau: int, max_m: int = 50) -> int:
    """Return the smallest m where FNN fraction drops low AND stabilizes.

    NOTE: For HAVOK, use optimal_m_havok() which adds margin for the linear model.
    This function returns the minimal embedding dimension (for attractor reconstruction)."""
    data = np.asarray(data).ravel()
    if len(data) < 20:
        return 3
    fracs = false_nearest_neighbors(data, tau, max_m=max_m)
    # Find where FNN drops below 5% AND stays there for 2+ consecutive dimensions
    for i in range(len(fracs) - 2):
        if all(f < 0.05 for f in fracs[i:i + 3]):
            return max(3, i + 1)
    # fallback sensible min
    return max(5, min(max(5, len(data) // 15), max_m))


def optimal_m_havok(data: np.ndarray, tau: int, max_m: int = 80) -> int:
    """SVD-spectrum estimate of HAVOK embedding dimension.

    Builds a Hankel at ``max_m``, computes the full SVD, finds the smallest
    m where cumulative singular-value energy reaches 99%, then applies an
    empirical safety margin.

    Heuristic (×3 multiplier)
    -------------------------
    The 99%-energy point identifies the dimension sufficient for attractor
    reconstruction (Takens).  HAVOK needs *more* dimensions for a good
    linear Koopman model, hence the ×3 margin.  This is an empirical
    rule of thumb validated on Lorenz, EEG, and financial benchmarks,
    NOT derived from theory.  Minimum enforced at 15.

    When it may fail
    ----------------
    - Quasi-periodic signals with slow SVD energy decay → overestimates m.
    - Records shorter than ~500 points → max_m must be reduced.

    Algorithm
    ---------
    Hankel at max_m → SVD → cumulative energy > 99% → m_99 × 3 → max(15, m)"""
    from havolib.embedding import hankel_matrix
    import numpy as np

    data = np.asarray(data).ravel()
    N = len(data)

    # Clamp max_m to be safe
    max_m = min(max_m, max(15, N // 10), 100)

    # Build Hankel at max_m and compute SVD
    try:
        H = hankel_matrix(data, max_m, tau)
        from scipy.linalg import svd
        _, s, _ = svd(H, full_matrices=False)

        # Cumulative energy
        cum_energy = np.cumsum(s**2) / np.sum(s**2)
        # Find where 99% energy is reached
        m_99 = int(np.searchsorted(cum_energy, 0.99)) + 1
        # Add margin: HAVOK needs more dimensions than minimal unfolding
        m_suggested = min(max_m, max(15, m_99 * 3))
    except Exception:
        m_suggested = max(15, min(50, N // 15))

    return int(m_suggested)


def suggest_parameters(data: np.ndarray, max_lag: int = 100, max_m: int = 50) -> dict:
    """
    Returns good tau (using Mutual Information) and m using improved FNN.
    """
    data = np.asarray(data).ravel()
    data = data - np.mean(data)

    tau = optimal_tau_mi(data, max_lag=max_lag)
    # Cap tau for HAVOK — MI picks good tau for attractor reconstruction,
    # but HAVOK's linear model needs smaller tau (coordinates should be somewhat
    # correlated so the linear model can leave meaningful residuals).
    tau = max(1, min(tau, max_lag // 5, 10))
    # Use SVD-spectrum method for HAVOK (FNN gives too few dimensions)
    m = optimal_m_havok(data, tau, max_m=max(50, max_m))

    return {
        "tau": int(tau),
        "m": int(m),
        "method": "mutual_information + SVD_spectrum",
        "recommendation": f"tau={tau}, m={m} (MI for tau, SVD spectrum for m)"
    }


if __name__ == "__main__":
    from data_loader import generate_lorenz
    _, x = generate_lorenz(n_points=1500)
    print(suggest_parameters(x))
