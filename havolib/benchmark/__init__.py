"""
Synthetic regime-shift datasets with known ground truth.

Each generator returns (t, x, shift_points) where shift_points is a list
of indices where the regime changes.
"""

import numpy as np
from typing import Tuple, List


def lorenz_with_shifts(
    n_points: int = 5000,
    shift_at: List[int] = None,
    dt: float = 0.01,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Lorenz attractor with optional parameter shifts (natural chaotic bursts)."""
    if shift_at is None:
        shift_at = []

    rng = np.random.default_rng(seed)
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    z = np.zeros(n_points)
    x[0], y[0], z[0] = 1.0, 1.0, 1.0

    shifts_detected = []
    current_rho = rho

    for i in range(1, n_points):
        if i in shift_at:
            current_rho = 28.0 + rng.uniform(-10, 10)  # shift rho
            shifts_detected.append(i)

        dx = sigma * (y[i - 1] - x[i - 1])
        dy = x[i - 1] * (current_rho - z[i - 1]) - y[i - 1]
        dz = x[i - 1] * y[i - 1] - beta * z[i - 1]
        x[i] = x[i - 1] + dt * dx
        y[i] = y[i - 1] + dt * dy
        z[i] = z[i - 1] + dt * dz

    t = np.arange(n_points) * dt
    return t, x, shifts_detected


def sinusoid_with_jump(
    n_points: int = 2000,
    jump_at: int = 1000,
    amplitude_before: float = 1.0,
    amplitude_after: float = 5.0,
    noise_std: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Sinusoid with an abrupt amplitude jump (clear regime shift)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 20 * np.pi, n_points)
    x = np.zeros(n_points)
    x[:jump_at] = amplitude_before * np.sin(t[:jump_at])
    x[jump_at:] = amplitude_after * np.sin(t[jump_at:])
    x += rng.normal(0, noise_std, n_points)
    shift_points = [jump_at] if jump_at < n_points else []
    return t, x, shift_points


def ar1_with_mean_shift(
    n_points: int = 2000,
    shift_at: int = 1000,
    phi: float = 0.7,
    mean_before: float = 0.0,
    mean_after: float = 2.0,
    noise_std: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """AR(1) process with a mean shift at shift_at."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n_points)
    eps = rng.normal(0, noise_std, n_points)

    mean = mean_before
    for i in range(1, n_points):
        if i == shift_at:
            mean = mean_after
        x[i] = mean + phi * (x[i - 1] - mean) + eps[i]

    return np.arange(n_points), x, [shift_at]


def logistic_map_bifurcation(
    n_points: int = 3000,
    shift_at: int = 1500,
    r_before: float = 3.5,
    r_after: float = 3.9,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Logistic map with a parameter change causing bifurcation (order → chaos)."""
    x = np.zeros(n_points)
    x[0] = 0.5

    r = r_before
    for i in range(1, n_points):
        if i == shift_at:
            r = r_after  # bifurcation: periodic → chaotic
        x[i] = r * x[i - 1] * (1 - x[i - 1])

    return np.arange(n_points), x, [shift_at]


def noise_to_oscillation(
    n_points: int = 2000,
    shift_at: int = 1000,
    noise_std: float = 0.5,
    osc_freq: float = 2.0,
    osc_amp: float = 3.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """White noise that transitions to a clear oscillation (emergence of order)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n_points)
    t = np.arange(n_points) / n_points * 20 * np.pi

    x[:shift_at] = rng.normal(0, noise_std, shift_at)
    x[shift_at:] = osc_amp * np.sin(osc_freq * t[shift_at:]) + rng.normal(0, noise_std * 0.5, n_points - shift_at)

    return t, x, [shift_at]


ALL_DATASETS = {
    "lorenz": (lorenz_with_shifts, "Lorenz attractor (chaotic bursts)"),
    "sinusoid_jump": (sinusoid_with_jump, "Sinusoid amplitude jump"),
    "ar1_shift": (ar1_with_mean_shift, "AR(1) mean shift"),
    "logistic_bifurcation": (logistic_map_bifurcation, "Logistic map bifurcation"),
    "noise_to_osc": (noise_to_oscillation, "Noise → oscillation emergence"),
}
