"""
Incremental Hankel Matrix — O(m) sliding-window construction.

Instead of rebuilding the full Hankel matrix each time (O(m²)),
we maintain a deque of the last N = (m-1)*tau + 1 values and
construct only the new row on each update.
"""

import numpy as np
from collections import deque
from typing import Optional


class IncrementalHankel:
    """Sliding-window Hankel matrix builder for streaming time series.

    Maintains the most recent row of the Hankel matrix for O(m) insertion.
    Full matrix reconstruction is O(m² * n_rows) and should only be done
    when the SVD needs to be fully recomputed (periodic recalibration).

    Usage:
        hankel = IncrementalHankel(m=50, tau=1)
        for x in stream:
            hankel.update(x)
            new_row = hankel.latest_row  # shape (m,)
    """

    def __init__(self, m: int, tau: int = 1, initial_data: Optional[np.ndarray] = None):
        if m < 2:
            raise ValueError("m must be >= 2")
        if tau < 1:
            raise ValueError("tau must be >= 1")

        self.m = m
        self.tau = tau
        self._stride = m * tau
        self._buffer = deque(maxlen=self._stride)

        self._latest_row: Optional[np.ndarray] = None

        if initial_data is not None:
            for v in initial_data:
                self.update(v)

    def update(self, value: float) -> None:
        """Push a new value. O(1) amortized."""
        self._buffer.append(value)
        if len(self._buffer) >= self._stride:
            # Build the newest row: take values at indices 0, tau, 2*tau, ..., (m-1)*tau
            # from the buffer (which is ordered oldest→newest)
            row = np.array([self._buffer[i * self.tau] for i in range(self.m)])
            self._latest_row = row

    @property
    def latest_row(self) -> Optional[np.ndarray]:
        """Most recent row of the Hankel matrix (shape (m,))."""
        return self._latest_row

    @property
    def ready(self) -> bool:
        """Whether enough data has been ingested to start producing rows."""
        return len(self._buffer) >= self._stride

    def full_matrix(self, n_rows: int) -> np.ndarray:
        """Reconstruct the last n_rows of the Hankel matrix. O(n_rows * m)."""
        if not self.ready:
            raise RuntimeError("Not enough data ingested yet")
        if n_rows > len(self._buffer) - self._stride + 1:
            n_rows = len(self._buffer) - self._stride + 1
        if n_rows < 1:
            raise ValueError("n_rows must be >= 1")

        data = list(self._buffer)
        N = len(data)
        H = np.zeros((n_rows, self.m))
        for i in range(n_rows):
            offset = N - self._stride - i
            for j in range(self.m):
                H[n_rows - 1 - i, j] = data[offset + j * self.tau]
        return H

    def get_series(self) -> np.ndarray:
        """Return the raw series buffer as numpy array."""
        return np.array(list(self._buffer))
