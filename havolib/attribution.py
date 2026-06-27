"""
Forcing Attribution — explain WHY HAVOK detects a regime shift.

Answers: "Which part of the signal caused this forcing spike?"
Uses SHAP-like perturbation analysis on the forcing extraction pipeline.

Usage:
    from havolib.attribution import explain_forcing_spike
    explanation = explain_forcing_spike(data, spike_index=1500)
    print(explanation)  # "Forcing spike at t=1500 caused by frequency shift
                        #  in channels [3,7] with contribution 73%"
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger("havok.attribution")


def explain_forcing_spike(
    X: np.ndarray,
    spike_index: int,
    tau: int = 1,
    m: int = 50,
    r: int = 5,
    context_window: int = 200,
) -> Dict:
    """Explain what caused a forcing spike at a specific time index.

    Performs perturbation analysis:
    1. Extract the segment around the spike
    2. Perturb different frequency bands / amplitude features
    3. Measure change in forcing magnitude
    4. Rank contributions

    Args:
        X: univariate time series
        spike_index: index of the forcing spike to explain
        tau, m, r: HAVOK parameters
        context_window: samples before/after spike to analyze

    Returns:
        dict with: cause (str), contributions (dict), top_contributor (str),
                   confidence (float)
    """
    X = np.asarray(X).ravel()
    n = len(X)

    start = max(0, spike_index - context_window)
    end = min(n, spike_index + context_window)
    x_seg = X[start:end]
    t_seg = np.arange(len(x_seg), dtype=float)

    # Baseline forcing
    from havolib.pipeline import HavokPipeline
    pipe = HavokPipeline(tau=tau, m=m, r=r)
    pipe.fit(t_seg, x_seg)
    forcing_base = pipe.get_forcing()
    spike_local = spike_index - start
    if spike_local < 0 or spike_local >= len(forcing_base):
        return {"cause": "Spike outside forcing range", "contributions": {}, "confidence": 0.0}
    base_spike = abs(forcing_base[spike_local])

    # Attributes to test
    attributes = {
        "amplitude_jump": lambda x: x - np.mean(x[:len(x)//2]) + np.mean(x[len(x)//2:]),
        "frequency_shift": lambda x: _add_chirp(x, 0.5, 2.0),
        "variance_burst": lambda x: x + np.random.randn(len(x)) * np.std(x) * 0.5,
        "trend_change": lambda x: x + np.linspace(0, np.std(x) * 2, len(x)),
        "noise_suppression": lambda x: _lowpass(x, cutoff=0.3),
    }

    contributions = {}
    for attr_name, perturb_fn in attributes.items():
        try:
            x_pert = perturb_fn(x_seg.copy())
            pipe2 = HavokPipeline(tau=tau, m=m, r=r)
            pipe2.fit(t_seg, x_pert)
            f_pert = pipe2.get_forcing()
            if spike_local < len(f_pert):
                pert_spike = abs(f_pert[spike_local])
                delta = base_spike - pert_spike  # positive = this attribute explains the spike
                contributions[attr_name] = max(0.0, float(delta))
        except Exception:
            contributions[attr_name] = 0.0

    total = sum(contributions.values()) + 1e-12
    contributions = {k: v / total for k, v in contributions.items()}

    # Find top contributor
    if contributions:
        top = max(contributions, key=contributions.get)
        top_pct = contributions[top] * 100
    else:
        top = "unknown"
        top_pct = 0

    # Confidence: how much of the spike is explained
    explained = sum(contributions.values())
    confidence = min(1.0, explained / max(base_spike, 1e-12))

    # Natural language explanation
    if top_pct > 50:
        cause = f"Forcing spike at t≈{spike_index} is primarily driven by {_human_name(top)} ({top_pct:.0f}% contribution)"
    elif top_pct > 25:
        runner_up = sorted(contributions, key=contributions.get, reverse=True)[1] if len(contributions) > 1 else "unknown"
        cause = f"Forcing spike at t≈{spike_index} caused by a mix of {_human_name(top)} ({top_pct:.0f}%) and {_human_name(runner_up)}"
    else:
        cause = f"Forcing spike at t≈{spike_index} has no single dominant cause — complex interaction detected"

    return {
        "cause": cause,
        "contributions": contributions,
        "top_contributor": top,
        "top_contribution_pct": top_pct,
        "confidence": confidence,
        "baseline_spike": float(base_spike),
    }


def _human_name(key: str) -> str:
    names = {
        "amplitude_jump": "amplitude jump",
        "frequency_shift": "frequency shift",
        "variance_burst": "variance burst",
        "trend_change": "trend change",
        "noise_suppression": "noise suppression",
    }
    return names.get(key, key)


def _add_chirp(x: np.ndarray, f0: float, f1: float) -> np.ndarray:
    t = np.linspace(0, 1, len(x))
    chirp = np.sin(2 * np.pi * (f0 + (f1 - f0) * t) * len(x) * t / 10)
    return x + chirp * np.std(x) * 0.3


def _lowpass(x: np.ndarray, cutoff: float = 0.3) -> np.ndarray:
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x))
    fft[freqs > cutoff * 0.5] *= 0.1
    return np.fft.irfft(fft, n=len(x))
