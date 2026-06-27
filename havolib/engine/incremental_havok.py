"""
Incremental HAVOK — streaming forcing extraction.

Uses batch HAVOK on sliding windows for simplicity and correctness.
Recomputes forcing every `stride` points from the latest N=hankel_window values.
O(N * m²) per batch, amortized O(m²) per point with large stride.
"""

import numpy as np
import logging
from numpy.linalg import lstsq
from typing import Optional, Tuple

from .ring_buffer import RingBuffer

logger = logging.getLogger("havok.engine.incremental")


class IncrementalHAVOK:
    """Streaming HAVOK regime-shift detector using periodic batch recomputation.

    Usage:
        havok = IncrementalHAVOK(m=50, tau=1, r=5)
        for x in data_stream:
            forcing, risk = havok.update(x)
    """

    def __init__(
        self,
        m: int = 50,
        tau: int = 1,
        r: int = 5,
        threshold_std: float = 3.0,
        window: int = 100,
        batch_stride: int = 20,
    ):
        self.m = m
        self.tau = tau
        self.r = r
        self.threshold_std = threshold_std
        self.window = window
        self.batch_stride = batch_stride

        # Buffers
        hankel_len = (m - 1) * tau + 1
        self._value_buffer = RingBuffer(capacity=max(hankel_len + batch_stride * 5, 5000))
        self._forcing_buffer = RingBuffer(capacity=max(window * 4, 2000))

        self._count = 0
        self._last_forcing = 0.0
        self._last_risk = 0.0
        self._V_cache: Optional[np.ndarray] = None

    def update(self, value: float) -> Tuple[float, float]:
        """Process one new data point. Returns (forcing, risk_score)."""
        self._value_buffer.push(value)
        self._count += 1

        # Only recompute periodically
        hankel_len = (self.m - 1) * self.tau + 1
        if self._count < hankel_len:
            return 0.0, 0.0

        if self._count % self.batch_stride == 0:
            self._recompute_forcing()

        # Compute risk from forcing buffer
        risk = self._compute_risk()

        return self._last_forcing, risk

    def _recompute_forcing(self) -> None:
        """Recompute forcing signal from the value buffer using batch HAVOK."""
        try:
            x = self._value_buffer.view(min(self._value_buffer.size, 4000))
            if len(x) < (self.m - 1) * self.tau + 4:
                return

            # Build Hankel matrix from recent values
            from havolib.embedding import hankel_matrix
            from scipy.linalg import svd

            H = hankel_matrix(x, self.m, self.tau)
            if H.shape[0] < self.r + 1:
                return

            # SVD
            U, s, Vt = svd(H, full_matrices=False)
            V = U[:, :self.r]  # eigen-time-delay coords (n_rows, r)
            self._V_cache = V

            # Forcing extraction
            if V.shape[0] >= 3 and V.shape[1] >= 2:
                dv = np.gradient(V, axis=0)
                X = np.column_stack([V[:, :-1], np.ones(V.shape[0])])
                y = dv[:, -1]
                coeffs, _, _, _ = lstsq(X, y, rcond=None)
                y_pred = X @ coeffs
                forcing_full = y - y_pred

                # Push forcing values to buffer
                for f in forcing_full[-self.batch_stride:]:
                    self._forcing_buffer.push(float(f))

                self._last_forcing = float(forcing_full[-1])
        except Exception as e:
            logger.warning(f"Batch SVD/Hankel recompute failed: {e} — forcing will be stale until next successful batch")  # not enough data yet

    def _compute_risk(self) -> float:
        """Compute risk score from forcing buffer."""
        if self._forcing_buffer.size < max(10, self.window // 4):
            return 0.0

        recent = self._forcing_buffer.view(min(self._forcing_buffer.size, self.window))
        abs_f = np.abs(recent)
        rolling_std = np.std(abs_f)
        if rolling_std < 1e-12:
            return 0.0

        current = abs(abs_f[-1])
        threshold = self.threshold_std * rolling_std
        if current < threshold:
            return 0.0

        risk = min(1.0, (current / max(threshold, 1e-12) - 1.0) / 3.0)
        return float(risk)

    def get_forcing_history(self, n: Optional[int] = None) -> np.ndarray:
        return self._forcing_buffer.view(n)

    def get_value_history(self, n: Optional[int] = None) -> np.ndarray:
        return self._value_buffer.view(n)

    @property
    def eigen_coordinates(self) -> Optional[np.ndarray]:
        return self._V_cache

    @property
    def point_count(self) -> int:
        return self._count
