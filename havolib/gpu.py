"""
GPU-accelerated linear algebra for HAVOK.

Transparently uses CuPy when a GPU is available, falls back to NumPy/SciPy.
Single import point: `from havolib.gpu import svd, lstsq, norm`

Design: zero-config — if cupy is installed and a GPU is detected, use it.
Otherwise NumPy. No user intervention needed.
"""

import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger("havok.gpu")

_GPU_AVAILABLE: Optional[bool] = None
_xp = np  # active array module (numpy or cupy)


def _detect_gpu() -> bool:
    """Detect if CuPy with a CUDA GPU is available."""
    global _GPU_AVAILABLE, _xp
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    try:
        import cupy as cp
        _ = cp.cuda.runtime.getDeviceCount()
        if cp.cuda.runtime.getDeviceCount() > 0:
            _xp = cp
            _GPU_AVAILABLE = True
            logger.info("GPU detected — using CuPy for linear algebra")
            return True
    except (ImportError, Exception):
        pass

    _GPU_AVAILABLE = False
    logger.info("No GPU detected — using NumPy/SciPy")
    return False


def is_gpu_available() -> bool:
    return _detect_gpu()


def get_array_module():
    """Return the active array module (numpy or cupy)."""
    _detect_gpu()
    return _xp


def svd(H: np.ndarray, full_matrices: bool = False, r: Optional[int] = None):
    """Truncated SVD — GPU if available.

    Args:
        H: input matrix (N, m)
        full_matrices: if True, return full U, Vt
        r: if provided, return only top r components

    Returns:
        U, s, Vt — all as numpy arrays (even if computed on GPU)
    """
    _detect_gpu()
    if _GPU_AVAILABLE:
        H_gpu = _xp.asarray(H)
        U_gpu, s_gpu, Vt_gpu = _xp.linalg.svd(H_gpu, full_matrices=full_matrices)
        U, s, Vt = U_gpu.get(), s_gpu.get(), Vt_gpu.get()
    else:
        from scipy.linalg import svd as scipy_svd
        U, s, Vt = scipy_svd(H, full_matrices=full_matrices)

    if r is not None:
        U, s, Vt = U[:, :r], s[:r], Vt[:r, :]

    return U, s, Vt


def lstsq(A: np.ndarray, b: np.ndarray, rcond: Optional[float] = None):
    """Least-squares — GPU if available."""
    _detect_gpu()
    if _GPU_AVAILABLE:
        A_gpu = _xp.asarray(A)
        b_gpu = _xp.asarray(b)
        x_gpu, residuals_gpu, rank_gpu, s_gpu = _xp.linalg.lstsq(A_gpu, b_gpu, rcond=rcond)
        return x_gpu.get(), residuals_gpu.get() if residuals_gpu.size else residuals_gpu, int(rank_gpu), s_gpu.get()
    else:
        from numpy.linalg import lstsq as np_lstsq
        return np_lstsq(A, b, rcond=rcond)


def norm(x: np.ndarray) -> float:
    """Vector 2-norm — GPU if available."""
    _detect_gpu()
    if _GPU_AVAILABLE:
        return float(_xp.linalg.norm(_xp.asarray(x)).get().item())
    return float(np.linalg.norm(x))


def eigvals(W: np.ndarray) -> np.ndarray:
    """Eigenvalues — GPU if available."""
    _detect_gpu()
    if _GPU_AVAILABLE:
        return _xp.linalg.eigvals(_xp.asarray(W)).get()
    return np.linalg.eigvals(W)
