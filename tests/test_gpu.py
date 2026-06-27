"""Tests for GPU module — CPU fallback, detection, CuPy compatibility."""
import pytest, numpy as np
from havolib.gpu import is_gpu_available, svd, lstsq, norm, eigvals

class TestGPUFallback:
    def test_is_gpu_available_returns_bool(self):
        assert isinstance(is_gpu_available(), bool)

    def test_svd_works_on_cpu(self):
        H = np.random.randn(50, 10)
        U, s, Vt = svd(H, r=3)
        assert U.shape == (50, 3)
        assert len(s) == 3
        assert Vt.shape == (3, 10)
        # U should be orthonormal
        assert np.allclose(U.T @ U, np.eye(3), atol=1e-6)

    def test_svd_full(self):
        H = np.random.randn(30, 8)
        U, s, Vt = svd(H, full_matrices=False)
        assert U.shape[0] == 30
        assert len(s) == min(30, 8)

    def test_lstsq_works_on_cpu(self):
        A = np.random.randn(20, 5)
        b = np.random.randn(20)
        x, residuals, rank, s = lstsq(A, b)
        assert x.shape == (5,)
        assert rank > 0

    def test_norm_works(self):
        x = np.array([3.0, 4.0])
        assert abs(norm(x) - 5.0) < 1e-10

    def test_eigvals_works(self):
        W = np.random.randn(5, 5)
        eigs = eigvals(W)
        assert len(eigs) == 5

    def test_svd_zero_matrix(self):
        H = np.zeros((20, 10))
        U, s, Vt = svd(H, r=3)
        assert np.allclose(s, 0)

    def test_svd_square(self):
        H = np.eye(10)
        U, s, Vt = svd(H, r=5)
        assert np.allclose(s[:5], 1.0)

    def test_lstsq_rank_deficient(self):
        A = np.column_stack([np.ones(20), np.ones(20)])  # rank 1
        b = np.random.randn(20)
        x, _, rank, _ = lstsq(A, b)
        assert len(x) == 2
