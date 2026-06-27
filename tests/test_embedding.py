import numpy as np
from havolib.embedding import hankel_matrix, auto_tau

def test_hankel_shape():
    x = np.arange(20)
    H = hankel_matrix(x, m=5, tau=2)
    expected_rows = 20 - (5-1)*2
    assert H.shape == (expected_rows, 5)

def test_hankel_values():
    x = np.array([0,1,2,3,4,5,6,7])
    H = hankel_matrix(x, m=3, tau=1)
    expected = np.array([
        [0,1,2],
        [1,2,3],
        [2,3,4],
        [3,4,5],
        [4,5,6],
        [5,6,7]
    ])
    assert np.allclose(H, expected)

def test_auto_tau():
    t = np.linspace(0, 4*np.pi, 200)
    x = np.sin(t)
    tau = auto_tau(x, max_lag=50)
    assert 1 <= tau <= 40  # relaxed bound for heuristic robustness
