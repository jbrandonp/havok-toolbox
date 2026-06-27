"""
Ring Buffer — fixed-size circular buffer backed by numpy.

O(1) push, O(1) slice access, zero-copy view of recent window.
Optimized for single-producer real-time streaming.
"""

import numpy as np
from typing import Optional


class RingBuffer:
    """Fixed-capacity circular buffer for streaming time series data.

    Usage:
        buf = RingBuffer(capacity=10000)
        for x in sensor_stream:
            buf.push(x)
            recent = buf.view(100)  # last 100 points, zero-copy
    """

    def __init__(self, capacity: int, dtype: type = np.float64):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._data = np.zeros(capacity, dtype=dtype)
        self._head = 0   # write position (next slot to fill)
        self._size = 0   # number of valid elements (≤ capacity)

    def push(self, value: float) -> None:
        self._data[self._head] = value
        self._head = (self._head + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def push_many(self, values: np.ndarray) -> None:
        n = len(values)
        if n >= self._capacity:
            values = values[-self._capacity:]
            self._data[:] = values
            self._size = self._capacity
            self._head = 0
            return
        remaining = self._capacity - self._head
        if n <= remaining:
            self._data[self._head:self._head + n] = values
            self._head += n
        else:
            first_chunk = remaining
            self._data[self._head:] = values[:first_chunk]
            second_chunk = n - first_chunk
            self._data[:second_chunk] = values[first_chunk:]
            self._head = second_chunk
        self._size = min(self._capacity, self._size + n)

    def view(self, n: Optional[int] = None) -> np.ndarray:
        if n is None or n > self._size:
            n = self._size
        if n == 0:
            return np.array([], dtype=self._data.dtype)
        start = (self._head - n) % self._capacity
        if start + n <= self._capacity:
            return self._data[start:start + n]
        else:
            first = self._capacity - start
            result = np.empty(n, dtype=self._data.dtype)
            result[:first] = self._data[start:]
            result[first:] = self._data[:n - first]
            return result

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_full(self) -> bool:
        return self._size == self._capacity

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> float:
        if idx < 0:
            idx = self._size + idx
        if idx < 0 or idx >= self._size:
            raise IndexError(f"index {idx} out of range [0, {self._size})")
        actual = (self._head - self._size + idx) % self._capacity
        return float(self._data[actual])
