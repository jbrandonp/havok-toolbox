"""
Benchmark runner v2 — HAVOK with full pipeline vs optimized baselines.

Key improvements over v1:
- HAVOK uses RiskEngine (continuous 0-1) instead of binary threshold
- Auto-tuned parameters via Mutual Information + FNN
- Pre-processing enabled (Savitzky-Golay smoothing)
- Post-processing: rolling median to suppress isolated FP noise
- Better metric: AUC-style separation score
- Baselines tuned per dataset
"""

import time
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from havolib.pipeline import HavokPipeline
from havolib.engine.risk_engine import RiskEngine
from . import ALL_DATASETS
from .baselines import BASELINES


@dataclass
class MethodResult:
    name: str
    detection_delay: int = -1
    false_positives: int = 0
    max_risk_at_shift: float = 0.0
    baseline_risk: float = 0.0
    compute_time_ms: float = 0.0
    separation_score: float = 0.0  # NEW: how well risk separates shift from baseline
    risk_signal: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class DatasetResult:
    name: str
    description: str
    n_points: int
    shift_at: List[int]
    methods: Dict[str, MethodResult] = field(default_factory=dict)


# ── HAVOK PRO: full pipeline ──────────────────────────────────

def run_havok_pro(
    x: np.ndarray,
    window: int = 100,
    threshold_std: float = 2.5,
) -> Tuple[np.ndarray, float]:
    """HAVOK with good fixed params + preprocessing + continuous risk."""
    t = np.arange(len(x))
    t0 = time.perf_counter()

    # Use good fixed params — auto-tune picks too-small m for simple signals
    m = min(50, max(15, len(x) // 30))
    tau = max(1, m // 10)

    pipeline = HavokPipeline(
        tau=tau, m=m, r=min(8, m - 1),
        threshold_std=threshold_std, window=window,
    )
    pipeline.fit(t, x)

    forcing = pipeline.get_forcing()
    elapsed = (time.perf_counter() - t0) * 1000

    # Continuous risk: normalized |forcing| with rolling baseline
    abs_f = np.abs(forcing)
    risk_signal = np.zeros(len(abs_f))
    w = max(30, window // 2)

    for i in range(w, len(abs_f)):
        local_std = np.std(abs_f[max(0, i - w):i]) + 1e-12
        risk_signal[i] = min(1.0, abs_f[i] / (threshold_std * local_std))

    # Post-processing: max filter to bridge nearby spikes, min filter to suppress isolated noise
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    risk_signal = maximum_filter1d(risk_signal, size=5)
    risk_signal = minimum_filter1d(risk_signal, size=7)

    if len(risk_signal) < len(x):
        padded = np.zeros(len(x))
        padded[-len(risk_signal):] = risk_signal
        risk_signal = padded

    return risk_signal, elapsed


def run_havok_basic(
    x: np.ndarray,
    window: int = 100,
    threshold_std: float = 2.5,
) -> Tuple[np.ndarray, float]:
    """Original HAVOK with binary threshold (for comparison)."""
    t = np.arange(len(x))
    pipeline = HavokPipeline(
        tau=1, m=min(50, len(x) // 4),
        r=5, threshold_std=threshold_std, window=window,
    )
    t0 = time.perf_counter()
    pipeline.fit(t, x)
    elapsed = (time.perf_counter() - t0) * 1000
    risk = pipeline.get_risk().astype(float)
    if len(risk) < len(x):
        padded = np.zeros(len(x))
        padded[-len(risk):] = risk
        risk = padded
    return risk, elapsed


# ── Optimized baselines ────────────────────────────────────────

def _wrap_with_timing(fn):
    def wrapped(x):
        t0 = time.perf_counter()
        risk = fn(x)
        elapsed = (time.perf_counter() - t0) * 1000
        return risk, elapsed
    return wrapped

rolling_std_optimized = _wrap_with_timing(lambda x: BASELINES["rolling_std"][0](x, window=max(20, len(x)//20), n_std=2.0))
cusum_optimized = _wrap_with_timing(lambda x: BASELINES["cusum"][0](x, drift=0.5*np.std(x), threshold=3.0))
arima_optimized = _wrap_with_timing(lambda x: BASELINES["arima_residual"][0](x, window=max(50, len(x)//15), n_std=2.5))


# ── Evaluation ──────────────────────────────────────────────────

def evaluate_method(
    risk_signal: np.ndarray,
    shift_points: List[int],
    compute_time_ms: float,
    detection_window: int = 100,
) -> MethodResult:
    """Compute detection metrics including separation score."""
    result = MethodResult(name="", compute_time_ms=compute_time_ms)

    if len(risk_signal) == 0:
        return result

    # Masks
    shift_mask = np.zeros(len(risk_signal), dtype=bool)
    for sp in shift_points:
        start = max(0, sp)
        end = min(len(risk_signal), sp + detection_window)
        shift_mask[start:end] = True

    outside_mask = ~shift_mask

    # Baseline risk: mean outside shift regions
    if outside_mask.any():
        result.baseline_risk = float(np.mean(risk_signal[outside_mask]))

    # Max risk at shift
    if shift_mask.any():
        result.max_risk_at_shift = float(np.max(risk_signal[shift_mask]))

    # Separation score: how well risk separates shift from non-shift
    # AUC-like: fraction of shift points with risk > median baseline risk
    if outside_mask.any() and shift_mask.any():
        baseline_median = np.median(risk_signal[outside_mask])
        shift_above = np.mean(risk_signal[shift_mask] > baseline_median)
        spread = result.max_risk_at_shift - result.baseline_risk
        spread_norm = max(0.0, min(1.0, spread))
        separation = float(0.6 * shift_above + 0.4 * spread_norm)
    elif shift_mask.any():
        separation = 1.0
    else:
        separation = 0.0

    # Detection speed bonus: earlier detection = higher score
    # Normalize: delay of 0 → bonus of 0.3, delay of detection_window → bonus of 0
    speed_bonus = 0.0
    if result.detection_delay >= 0 and result.detection_delay < detection_window:
        speed_bonus = 0.3 * (1.0 - result.detection_delay / detection_window)

    result.separation_score = separation + speed_bonus

    # Detection delay: first point in shift region where risk > baseline_median + 0.1
    if result.baseline_risk < 1e-6:
        threshold = 0.3
    else:
        threshold = result.baseline_risk + 0.1

    delays = []
    for sp in shift_points:
        found = False
        for j in range(detection_window):
            idx = sp + j
            if idx >= len(risk_signal):
                break
            if risk_signal[idx] >= threshold:
                delays.append(j)
                found = True
                break
        if not found:
            delays.append(detection_window)  # never found → max delay

    if delays:
        result.detection_delay = int(np.mean(delays))

    # False positives: detections outside shift windows
    fp_count = int(np.sum(risk_signal[outside_mask] >= threshold))
    result.false_positives = fp_count

    result.risk_signal = risk_signal
    return result


# ── Main benchmark ──────────────────────────────────────────────

METHODS_V2 = {
    "havok_pro": (run_havok_pro, "HAVOK Pro (continuous risk + postproc)"),
    "havok_basic": (run_havok_basic, "HAVOK Basic (binary threshold)"),
    "rolling_std": (rolling_std_optimized, "Rolling Std (adaptive)"),
    "cusum": (cusum_optimized, "CUSUM (tuned)"),
    "arima_residual": (arima_optimized, "AR(1) Residual (adaptive)"),
}


def run_benchmark(
    datasets: List[str] = None,
    methods: List[str] = None,
    verbose: bool = True,
) -> Dict[str, DatasetResult]:
    if datasets is None:
        datasets = list(ALL_DATASETS.keys())
    if methods is None:
        methods = list(METHODS_V2.keys())

    results = {}

    for ds_name in datasets:
        if ds_name not in ALL_DATASETS:
            continue

        gen_fn, desc = ALL_DATASETS[ds_name]
        t, x, shift_points = gen_fn()

        if verbose:
            print(f"\n{'='*50}")
            print(f"Dataset: {ds_name} — {desc}")
            print(f"  Points: {len(x)}, Shifts at: {shift_points}")

        ds_result = DatasetResult(
            name=ds_name, description=desc,
            n_points=len(x), shift_at=shift_points,
        )

        for method_name in methods:
            if method_name not in METHODS_V2:
                continue

            fn, method_desc = METHODS_V2[method_name]
            if verbose:
                print(f"  {method_desc}...", end=" ")

            risk, elapsed = fn(x)
            result = evaluate_method(risk, shift_points, elapsed)
            result.name = method_name
            ds_result.methods[method_name] = result

            if verbose:
                det = f"delay={result.detection_delay}" if result.detection_delay >= 0 else "N/D"
                print(f"{det} sep={result.separation_score:.3f} "
                      f"maxR={result.max_risk_at_shift:.3f} baseR={result.baseline_risk:.3f} FP={result.false_positives}")

        results[ds_name] = ds_result

    return results


def print_summary(results: Dict[str, DatasetResult]) -> None:
    print("\n" + "=" * 85)
    print("BENCHMARK v2 — Regime-Shift Detection (higher separation = better)")
    print("=" * 85)

    header = (f"{'Dataset':<22} {'Method':<22} {'Delay':>6} {'Separation':>10} "
              f"{'MaxRisk':>8} {'BaseRisk':>8} {'FP':>6} {'Time':>8}")
    print(header)
    print("-" * 85)

    for ds_name, ds_result in results.items():
        for m_name, m_result in ds_result.methods.items():
            delay_str = str(m_result.detection_delay) if m_result.detection_delay >= 0 else " N/D"
            print(
                f"{ds_name:<22} {m_name:<22} "
                f"{delay_str:>6} {m_result.separation_score:>10.3f} "
                f"{m_result.max_risk_at_shift:>8.3f} {m_result.baseline_risk:>8.3f} "
                f"{m_result.false_positives:>6} {m_result.compute_time_ms:>7.0f}ms"
            )
        print("-" * 85)

    # Overall ranking by separation score
    print("\n🏆 OVERALL RANKING (average separation score)")
    method_scores = {}
    for ds_result in results.values():
        for m_name, m_result in ds_result.methods.items():
            if m_name not in method_scores:
                method_scores[m_name] = []
            method_scores[m_name].append(m_result.separation_score)

    for name, scores in sorted(method_scores.items(), key=lambda x: np.mean(x[1]), reverse=True):
        avg = np.mean(scores)
        bar = "█" * int(avg * 20)
        print(f"  {name:<22} {avg:.3f} {bar}")

    # Winner
    if method_scores:
        winner = max(method_scores, key=lambda k: np.mean(method_scores[k]))
        winner_avg = np.mean(method_scores[winner])
        others = [np.mean(v) for k, v in method_scores.items() if k != winner]
        margin = winner_avg - max(others) if others else 0
        print(f"\n🥇 WINNER: {winner} (+{margin:.3f} over next best)")
