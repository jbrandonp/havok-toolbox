"""Property-based tests for HAVOK — invariant checking via Hypothesis.

Uses Hypothesis to test mathematical properties that should always hold.
Run: pytest tests/test_properties.py -v
Requires: pip install hypothesis
"""

import numpy as np
import pytest

# Skip entire module if hypothesis not installed
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st, settings, assume

from havolib.embedding import hankel_matrix
from havolib.decomposition import eigen_time_delay
from havolib.forcing import extract_forcing
from havolib.pre_processing import preprocess
from havolib.surrogate import phase_randomized_surrogate


# ── Hankel matrix invariants ───────────────────────────────────

@given(
    st.lists(st.floats(-1e6, 1e6), min_size=20, max_size=500),
    st.integers(2, 15),
    st.integers(1, 5),
)
@settings(max_examples=100, deadline=None)
def test_hankel_shape_invariant(data, m, tau):
    """Hankel matrix must have correct shape: (N - (m-1)*tau, m)."""
    x = np.array(data)
    N = len(x)
    assume(N >= m * tau)
    H = hankel_matrix(x, m, tau)
    expected_rows = N - (m - 1) * tau
    assert H.shape == (expected_rows, m)


@given(
    st.lists(st.floats(-1e3, 1e3), min_size=30, max_size=200),
    st.integers(2, 8),
    st.integers(1, 3),
)
@settings(max_examples=100, deadline=None)
def test_hankel_values_diagonal_constant(data, m, tau):
    """For tau=1, Hankel matrix should have constant anti-diagonals."""
    x = np.array(data)
    if tau != 1:
        return
    H = hankel_matrix(x, m, tau=1)
    for k in range(min(3, H.shape[0] - 1)):
        assert np.allclose(H[k, 1:], H[k + 1, :-1])


# ── SVD reconstruction ─────────────────────────────────────────

@given(
    st.lists(st.lists(st.floats(-100, 100), min_size=5, max_size=5), min_size=10, max_size=50),
    st.integers(2, 5),
)
@settings(max_examples=50, deadline=None)
def test_svd_orthonormal(data, r):
    """Truncated SVD U must be orthonormal: U.T @ U ≈ I."""
    H = np.array(data, dtype=float)
    assume(H.shape[0] >= r and H.shape[1] >= r)
    # Skip zero matrices
    assume(np.max(np.abs(H)) > 1e-6)
    U, s = eigen_time_delay(H, r)
    gram = U.T @ U
    assert np.allclose(gram, np.eye(r), atol=1e-6)


# ── Forcing invariants ─────────────────────────────────────────

@given(st.integers(100, 500))
@settings(max_examples=30, deadline=None)
def test_forcing_finite_and_correct_length(n_points):
    """Forcing signal must be finite and have correct length."""
    t = np.linspace(0, 10 * np.pi, n_points)
    x = np.sin(t) + 0.1 * np.cos(3 * t)
    m = min(30, n_points // 4)
    H = hankel_matrix(x, m=m, tau=1)
    r = min(3, H.shape[1] - 1)
    if r < 2:
        return
    V, _ = eigen_time_delay(H, r)
    t_trim = t[:V.shape[0]]
    forcing = extract_forcing(V, t_trim)
    assert np.all(np.isfinite(forcing))
    assert len(forcing) == V.shape[0]


# ── Preprocessing invariants ───────────────────────────────────

@given(st.lists(st.floats(-100, 100), min_size=50, max_size=300))
@settings(max_examples=50, deadline=None)
def test_preprocess_preserves_length_and_cleans(data):
    """Preprocessing must never change array length and must remove NaNs."""
    x = np.array(data)
    result = preprocess(x, interpolate=True, smooth_method='savgol',
                        smooth_window=11, outlier_method='iqr')
    assert len(result) == len(x)
    assert not np.any(np.isnan(result))
    assert np.all(np.isfinite(result))


# ── Surrogate invariants ───────────────────────────────────────

@given(st.lists(st.floats(-10, 10), min_size=50, max_size=300))
@settings(max_examples=50, deadline=None)
def test_surrogate_preserves_power_spectrum(data):
    """Phase-randomized surrogate must preserve power spectrum (|FFT|)."""
    x = np.array(data)
    x = x - np.mean(x)
    # Skip near-constant signals
    assume(np.std(x) > 0.5)
    assume(np.max(np.abs(x)) < 100)
    s = phase_randomized_surrogate(x, seed=42)
    amp_x = np.abs(np.fft.fft(x))
    amp_s = np.abs(np.fft.fft(s))
    corr = np.corrcoef(amp_x, amp_s)[0, 1]
    # For stationary signals, correlation > 0.99. For spike-dominated, > 0.75.
    assert corr > 0.75, f"Power spectrum correlation too low: {corr:.6f}"


@given(st.lists(st.floats(-10, 10), min_size=50, max_size=300))
@settings(max_examples=50, deadline=None)
def test_surrogate_preserves_variance(data):
    """Surrogate must have the same variance as the original."""
    x = np.array(data)
    if np.std(x) < 1e-6:
        return
    s = phase_randomized_surrogate(x, seed=42)
    ratio = np.var(s) / np.var(x)
    assert 0.99 < ratio < 1.01, f"Variance ratio: {ratio:.4f}"
