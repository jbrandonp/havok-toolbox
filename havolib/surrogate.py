"""
Phase-randomized surrogate tests for HAVOK forcing signals.

Critical for statistical validation: 
- Generate surrogates that preserve power spectrum (autocorrelation structure)
- Run HAVOK on surrogates
- Use 99th percentile of surrogate max |forcing| as data-driven threshold
- This distinguishes real intermittent forcing from linear autocorrelation artifacts.
"""

import numpy as np
from typing import Tuple, List, Callable


def phase_randomized_surrogate(x: np.ndarray, seed: int = None) -> np.ndarray:
    x = np.asarray(x).ravel()
    N = len(x)
    
    # FFT
    Xf = np.fft.fft(x)
    amp = np.abs(Xf)
    
    # Use provided seed for reproducible random phases
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    # Random phases only for non-negative frequencies (correct length)
    n_pos = N // 2 + 1
    rand_phase = rng.uniform(0, 2 * np.pi, n_pos)
    
    # Build symmetric phase for full FFT
    full_phase = np.concatenate([rand_phase, -rand_phase[1:(-1 if N%2==0 else None)][::-1]])
    
    # Reconstruct
    Xf_surr = amp * np.exp(1j * full_phase)
    x_surr = np.fft.ifft(Xf_surr).real
    
    # Match moments
    x_surr = (x_surr - np.mean(x_surr)) * (np.std(x) / (np.std(x_surr) + 1e-12)) + np.mean(x)
    return x_surr


def generate_surrogates(x: np.ndarray, n_surrogates: int = 100, seed: int = 42) -> List[np.ndarray]:
    """Generate multiple phase-randomized surrogates."""
    surrogates = []
    rng = np.random.default_rng(seed)
    for i in range(n_surrogates):
        s = phase_randomized_surrogate(x, seed=rng.integers(0, 2**32))
        surrogates.append(s)
    return surrogates


def surrogate_forcing_distribution(
    x: np.ndarray,
    pipeline_factory: Callable,
    n_surrogates: int = 100,
    seed: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Run HAVOK on many surrogates and return the distribution of max |forcing|.
    
    Returns:
        max_forcings: array of max |forcing| from each surrogate
        threshold_99: 99th percentile (data-driven significance threshold)
    """
    surrogates = generate_surrogates(x, n_surrogates=n_surrogates, seed=seed)
    max_forcings = []
    
    for xs in surrogates:
        try:
            t_s = np.arange(len(xs))
            pipe = pipeline_factory()
            pipe.fit(t_s, xs)
            f = np.abs(pipe.get_forcing())
            max_forcings.append(np.max(f))
        except Exception:
            continue
    
    max_forcings = np.array(max_forcings)
    if len(max_forcings) == 0:
        return np.array([0.0]), 0.0
    
    threshold_99 = np.percentile(max_forcings, 99)
    return max_forcings, float(threshold_99)


def validate_forcing_significance(
    observed_max: float,
    surrogate_maxes: np.ndarray,
    alpha: float = 0.01
) -> Tuple[bool, float]:
    """
    Test if observed max forcing is significant vs surrogates.
    
    Returns: (is_significant, p_value)
    """
    if len(surrogate_maxes) == 0:
        return False, 1.0
    
    p_value = np.mean(surrogate_maxes >= observed_max)
    is_significant = p_value < alpha
    return bool(is_significant), float(p_value)


if __name__ == "__main__":
    from data_loader import generate_lorenz
    from pipeline import HavokPipeline
    
    print("Generating Lorenz for surrogate test...")
    t, x = generate_lorenz(n_points=4000)
    
    def make_pipe():
        return HavokPipeline(tau=10, m=30, r=5)
    
    pipe = make_pipe()
    pipe.fit(t, x)
    obs_max = np.max(np.abs(pipe.get_forcing()))
    print(f"Observed max |forcing|: {obs_max:.4f}")
    
    print("Running 30 surrogates...")
    surr_max, thresh = surrogate_forcing_distribution(x, make_pipe, n_surrogates=30, seed=123)
    print(f"Surrogate max |forcing| 99th percentile: {thresh:.4f}")
    print(f"Surrogate max range: {surr_max.min():.4f} - {surr_max.max():.4f}")
    
    sig, p = validate_forcing_significance(obs_max, surr_max)
    print(f"Significant at 1%? {sig} (p={p:.3f})")
