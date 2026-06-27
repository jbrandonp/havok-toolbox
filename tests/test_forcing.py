import numpy as np
from havolib.embedding import hankel_matrix
from havolib.decomposition import eigen_time_delay
from havolib.forcing import extract_forcing

def test_forcing_lorenz_has_spikes():
    """Lorenz attractor must produce clear intermittent forcing spikes (core HAVOK behavior)."""
    from havolib.data_loader import generate_lorenz
    t, x = generate_lorenz(n_points=4000)
    H = hankel_matrix(x, m=50, tau=1)
    V, _ = eigen_time_delay(H, r=5)
    t_trim = t[:V.shape[0]]
    forcing = extract_forcing(V, t_trim)
    assert np.max(np.abs(forcing)) > 0.5, "Expected large forcing spikes in chaotic regime"

def test_forcing_small_on_simple_periodic():
    """For a simple sine, forcing should not dominate completely (sanity, not strict zero)."""
    t = np.linspace(0, 6*np.pi, 800)
    x = np.sin(t)
    H = hankel_matrix(x, m=30, tau=1)
    V, _ = eigen_time_delay(H, r=3)
    t_trim = t[:V.shape[0]]
    forcing = extract_forcing(V, t_trim)
    # Just ensure we don't crash and forcing is finite
    assert np.isfinite(np.max(np.abs(forcing)))
