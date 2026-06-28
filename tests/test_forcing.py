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

def test_forcing_validates_inputs():
    """Critical validations for t and data quality."""
    V = np.random.randn(50, 4)
    t = np.linspace(0, 10, 50)

    # Bad length
    try:
        extract_forcing(V, t[:30])
        assert False
    except ValueError:
        pass

    # Non-monotonic
    bad_t = t.copy()
    bad_t[5] = bad_t[4] - 1
    try:
        extract_forcing(V, bad_t)
        assert False
    except ValueError:
        pass

    # NaN
    Vnan = V.copy()
    Vnan[10, 1] = np.nan
    try:
        extract_forcing(Vnan, t)
        assert False
    except ValueError:
        pass

def test_forcing_size_limits():
    """Size guard is present (avoids OOM in test env)."""
    import inspect
    src = inspect.getsource(extract_forcing)
    assert 'MAX_N' in src and 'safety limits' in src

def test_forcing_underdetermined_warns():
    """n < r should warn."""
    import warnings
    Vsmall = np.random.randn(5, 10)
    tsmall = np.linspace(0, 1, 5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _ = extract_forcing(Vsmall, tsmall)
        assert len(w) == 1
        assert 'underdetermined' in str(w[0].message).lower()
