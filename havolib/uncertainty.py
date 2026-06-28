"""
Uncertainty quantification for HAVOK (new dedicated module per upgrade spec).

Provides:
- Phase-randomized surrogates (Theiler)
- Block bootstrap
- Conformal-style intervals (light)
- CRPS for probabilistic evaluation
- Helpers used by estimator (fit_with_ci) and future forecasters.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, List, Optional

# Delegate to canonical implementation in surrogate (avoids duplication; keeps guards, logic in one place)
from .surrogate import phase_randomized_surrogate, generate_surrogates


def block_bootstrap(x: np.ndarray, n_boot: int = 100, block_size: Optional[int] = None, seed: int = 42) -> List[np.ndarray]:
    """Block bootstrap to preserve autocorrelation structure."""
    x = np.asarray(x).ravel()
    n = len(x)
    if block_size is None:
        block_size = max(10, int(np.sqrt(n)))
    n_blocks = n // block_size
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        boot = np.concatenate([x[i*block_size:(i+1)*block_size] for i in idx])[:n]
        if len(boot) < n:
            boot = np.pad(boot, (0, n - len(boot)), mode='edge')
        boots.append(boot)
    return boots


def crps(observations: np.ndarray, forecasts: np.ndarray) -> float:
    """Continuous Ranked Probability Score (lower is better)."""
    obs = np.asarray(observations).ravel()
    fc = np.asarray(forecasts).ravel()
    # Simple empirical CRPS
    term1 = np.mean(np.abs(fc - obs))
    term2 = np.mean(np.abs(fc[:, None] - fc[None, :])) / 2.0 if len(fc) > 1 else 0.0
    return float(term1 - term2)


def conformal_interval(residuals: np.ndarray, alpha: float = 0.1) -> Tuple[float, float]:
    """Simple conformal prediction quantile interval from residuals."""
    q = np.quantile(np.abs(residuals), 1 - alpha)
    return -q, q


# Convenience re-exports
__all__ = [
    "phase_randomized_surrogate",
    "generate_surrogates",
    "block_bootstrap",
    "crps",
    "conformal_interval",
]
