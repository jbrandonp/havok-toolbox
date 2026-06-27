"""
Brand SVD — incremental truncated SVD update.

Implements Brand (2002, "Incremental Singular Value Decomposition")
for efficiently updating the SVD when a single new row is added
to a data matrix. O(m * r) per update instead of O(m² * r) for full SVD.

Reference:
  Brand, M. (2002). Incremental singular value decomposition of
  uncertain data with missing values. ECCV 2002.
"""

import numpy as np
from scipy.linalg import svd
from typing import Tuple, Optional


class BrandSVD:
    """Incremental truncated SVD for streaming Hankel matrices.

    Maintains an approximate rank-r SVD of a growing (or sliding-window)
    matrix. When a new row arrives, the SVD is updated via the Brand
    algorithm in O(m * r²) time.

    Periodic full recomputation (every ~1000 updates) prevents drift.
    """

    def __init__(
        self,
        m: int,
        r: int,
        initial_matrix: Optional[np.ndarray] = None,
        recalibrate_every: int = 500,
    ):
        if r < 1:
            raise ValueError("r must be >= 1")
        if r > m:
            raise ValueError("r must be <= m")

        self.m = m
        self.r = r
        self.recalibrate_every = recalibrate_every

        # Current SVD: H ≈ U @ diag(s) @ Vt  with U.shape=(n_rows, r), V.shape=(m, r)
        self.V: Optional[np.ndarray] = None  # right singular vectors (m, r)
        self.s: Optional[np.ndarray] = None
        self._n_rows: int = 0
        self._updates_since_calibration = 0

        if initial_matrix is not None and initial_matrix.shape[0] >= r:
            self._full_recompute(initial_matrix)

    def _full_recompute(self, H: np.ndarray) -> None:
        """Full SVD recomputation (used for initialization and periodic recal)."""
        U, s, Vt = svd(H, full_matrices=False)
        self.V = Vt[:self.r, :].T  # (m, r) — right singular vectors
        self.s = s[:self.r]
        self._n_rows = H.shape[0]
        self._updates_since_calibration = 0

    def update(self, new_row: np.ndarray) -> None:
        """Update SVD with a single new row. O(m * r²).

        Args:
            new_row: shape (m,) — new row of the data matrix.
        """
        new_row = np.asarray(new_row, dtype=float).reshape(1, -1)
        if new_row.shape[1] != self.m:
            raise ValueError(f"Expected row of length {self.m}, got {new_row.shape[1]}")

        if self.V is None:
            # First update — accumulate until we have enough rows
            if not hasattr(self, '_pending'):
                self._pending = []
            self._pending.append(new_row.ravel())
            if len(self._pending) >= self.r:
                H = np.array(self._pending)
                self._full_recompute(H)
                del self._pending
            return

        # Brand update using right singular vectors V (m × r)
        # 1. Project: c = new_row @ V  (1 × r)
        c = new_row @ self.V  # (1, r)

        # 2. Residual: h = new_row - c @ V.T  (1 × m)
        h = new_row - c @ self.V.T  # (1, m)

        # 3. Residual norm for rank augmentation
        h_norm = np.linalg.norm(h)
        if h_norm > 1e-12:
            h_unit = h.T / h_norm  # (m, 1)
        else:
            h_unit = np.zeros((self.m, 1))

        # 4. Build (r+1) × (r+1) matrix to diagonalize
        K = np.zeros((self.r + 1, self.r + 1))
        K[:self.r, :self.r] = np.diag(self.s)
        K[:self.r, self.r] = c.ravel()
        K[self.r, self.r] = h_norm

        # 5. SVD of small matrix K
        Uk, sk, Vkt = svd(K, full_matrices=False)

        # 6. Truncate to top r
        self.s = sk[:self.r]

        # 7. Update V basis: V_new = [V, h_unit] @ Vk[:, :r]
        V_aug = np.hstack([self.V, h_unit])  # (m, r+1)
        self.V = V_aug @ Vkt[:self.r, :].T  # (m, r+1) @ (r+1, r) = (m, r)

        self._n_rows += 1
        self._updates_since_calibration += 1

    def get_eigen_coordinates(self) -> np.ndarray:
        """Return the right singular vectors V (m, r) for eigen-time-delay coordinates."""
        if self.V is None:
            raise RuntimeError("SVD not initialized yet")
        return self.V.copy()

    def get_singular_values(self) -> np.ndarray:
        """Return current singular values."""
        if self.s is None:
            raise RuntimeError("SVD not initialized yet")
        return self.s.copy()

    def needs_recalibration(self) -> bool:
        return self._updates_since_calibration >= self.recalibrate_every

    def recalibrate(self, H: np.ndarray) -> None:
        """Force full SVD recomputation on the given matrix."""
        self._full_recompute(H)
