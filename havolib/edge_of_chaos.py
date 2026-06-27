"""
Edge-of-chaos metrics for HAVOK forcing analysis.

Computes quantitative measures of how close a system is to the edge of chaos:
- Largest Lyapunov Exponent (LLE)
- Correlation Dimension (Grassberger-Procaccia)
- Critical Slowing Down (lag-1 autocorrelation)
- HAVOK-combined edge-of-chaos score
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from typing import Tuple


def largest_lyapunov_exponent(
    x: np.ndarray,
    tau: int = 1,
    m: int = 5,
    dt: float = 1.0,
    n_steps: int = 20,
) -> float:
    """
    Rosenstein algorithm for Largest Lyapunov Exponent.

    Positive LLE → chaos. Higher → more chaotic / further from order.
    Near-zero → edge of chaos / critical dynamics.

    Args:
        x: Time series
        tau: Delay for embedding
        m: Embedding dimension
        dt: Sampling interval
        n_steps: Number of divergence steps

    Returns:
        Largest Lyapunov Exponent (nats/time unit)
    """
    x = np.asarray(x).ravel()
    N = len(x)
    M = N - (m - 1) * tau

    if M < 50:
        return 0.0

    # Build delay vectors
    X = np.array([x[i:i + m * tau:tau] for i in range(M)])

    # Find nearest neighbors (excluding temporal neighbors)
    tree = cKDTree(X)
    min_sep = max(1, m * tau // 2)

    # Track divergence
    divergences = []

    for i in range(M - n_steps):
        # Find nearest neighbor not too close in time
        dists, idxs = tree.query(X[i], k=min(M, 20))
        nn_idx = None
        for j in idxs:
            if abs(i - j) > min_sep:
                nn_idx = j
                break

        if nn_idx is None:
            continue

        # Compute divergence over n_steps
        d0 = np.linalg.norm(X[i] - X[nn_idx])
        if d0 < 1e-12:
            continue

        div_curve = []
        for k in range(1, n_steps + 1):
            if i + k >= M or nn_idx + k >= M:
                break
            dk = np.linalg.norm(X[i + k] - X[nn_idx + k])
            div_curve.append(np.log(dk / d0))

        if len(div_curve) == n_steps:
            divergences.append(div_curve)

    if not divergences:
        return 0.0

    # Average over all reference points
    avg_div = np.mean(divergences, axis=0)
    steps = np.arange(1, n_steps + 1) * dt

    # Linear fit to get LLE
    if len(steps) > 1:
        slope, _ = np.polyfit(steps, avg_div, 1)
        return float(slope)

    return 0.0


def correlation_dimension(
    x: np.ndarray,
    tau: int = 1,
    m: int = 5,
    eps_range: Tuple[float, float] = (0.01, 1.0),
    n_eps: int = 20,
) -> float:
    """
    Grassberger-Procaccia correlation dimension.

    Low dimension → ordered. High dimension → chaotic.
    Values between ~2 and ~7 indicate chaos.

    Returns:
        Estimated correlation dimension
    """
    x = np.asarray(x).ravel()
    N = len(x)
    M = N - (m - 1) * tau

    if M < 100:
        return 0.0

    # Build delay vectors (subsample for performance)
    max_points = min(M, 2000)
    idx = np.linspace(0, M - 1, max_points, dtype=int)
    X = np.array([x[i:i + m * tau:tau] for i in idx])

    # Compute pairwise distances
    dists = pdist(X, metric='euclidean')

    # Compute correlation sum for different epsilons
    epsilons = np.logspace(np.log10(eps_range[0]), np.log10(eps_range[1]), n_eps)
    log_C = []

    for eps in epsilons:
        C = np.mean(dists < eps)
        log_C.append(np.log(max(C, 1e-12)))

    # Fit slope in scaling region
    log_eps = np.log(epsilons)
    # Use middle portion for robust fit
    mid_start = n_eps // 4
    mid_end = 3 * n_eps // 4

    if mid_end - mid_start > 2:
        slope, _ = np.polyfit(log_eps[mid_start:mid_end], log_C[mid_start:mid_end], 1)
        return float(slope)

    return 0.0


def critical_slowing_down(x: np.ndarray, lag: int = 1) -> float:
    """
    Lag-1 autocorrelation as indicator of critical slowing down.

    As system approaches tipping point, lag-1 autocorrelation → 1.0.
    This is the classic early-warning signal for regime shifts.

    Returns:
        Lag-1 autocorrelation (0 to 1)
    """
    x = np.asarray(x).ravel()
    if len(x) < lag + 2:
        return 0.0

    acf = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    return float(max(0.0, min(1.0, acf)))


def edge_of_chaos_score(
    x: np.ndarray,
    tau: int = None,
    m: int = 5,
    dt: float = 1.0,
) -> dict:
    """
    Combined edge-of-chaos score.

    High score → system is approaching the edge of chaos (critical dynamics).
    Low score → system is either ordered (rigid) or fully chaotic (random).

    Returns:
        dict with individual metrics and combined score
    """
    from havolib.auto_tune import optimal_tau_mi

    x = np.asarray(x).ravel()
    if tau is None:
        tau = max(1, optimal_tau_mi(x, max_lag=min(100, len(x) // 4)))

    lle = largest_lyapunov_exponent(x, tau=tau, m=m, dt=dt)
    csd = critical_slowing_down(x, lag=1)

    # Edge of chaos = system is NEITHER fully ordered NOR fully chaotic
    # Moderate LLE (> 0 but not too high) + high CSD → edge of chaos
    lle_normalized = 1.0 - np.exp(-abs(lle) * 5)  # 0=ordered, 1=chaotic

    # Combined: edge of chaos = moderate chaos + critical slowing down
    edge_score = (1.0 - abs(lle_normalized - 0.5) * 2.0) * csd
    edge_score = max(0.0, min(1.0, edge_score))

    return {
        "largest_lyapunov_exponent": lle,
        "critical_slowing_down_lag1": csd,
        "edge_of_chaos_score": edge_score,
        "interpretation": (
            "🔥 EDGE OF CHAOS — rich adaptive dynamics"
            if edge_score > 0.6
            else "🧊 ORDERED — rigid, predictable"
            if lle < 0.01
            else "🌪️ CHAOTIC — random, unstructured"
            if lle > 0.5
            else "⚡ TRANSITION — approaching the edge"
        ),
        "tau": tau,
        "m": m,
    }
