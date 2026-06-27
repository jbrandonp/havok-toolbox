"""
Adaptive Non-Stationary HAVOK — auto-adjusts parameters when dynamics drift.

This is the #1 moat feature from the strategic plan. It continuously monitors
the data distribution and retriggers HAVOK with updated parameters whenever
the underlying dynamics change significantly.

Key innovation: instead of fixed (tau, m, r), the algorithm detects
statistical regime changes in the raw signal and recomputes optimal
parameters for each new regime segment.

Usage:
    from havolib.adaptive import AdaptiveHAVOK
    adp = AdaptiveHAVOK()
    result = adp.fit_transform(long_time_series)  # auto-detects segments
    # result has per-segment forcing + transition points
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import logging
import warnings

logger = logging.getLogger("havok.adaptive")


@dataclass
class RegimeSegment:
    """One detected regime with its HAVOK analysis."""
    start_idx: int
    end_idx: int
    tau: int
    m: int
    r: int
    forcing: np.ndarray
    risk: np.ndarray
    max_forcing: float
    n_risk_events: int
    edge_score: float
    lyapunov: float

    @property
    def duration(self) -> int:
        return self.end_idx - self.start_idx


@dataclass
class AdaptiveResult:
    """Result of adaptive HAVOK analysis."""
    n_samples: int
    segments: List[RegimeSegment]
    transition_points: List[int]           # indices where regimes change
    full_forcing: np.ndarray               # concatenated forcing across all segments
    full_risk: np.ndarray                  # concatenated risk
    parameter_timeline: Dict[str, np.ndarray]  # how parameters evolved over time

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"Adaptive HAVOK Analysis — {len(self.segments)} regimes detected",
            "=" * 60,
            f"  Transitions at: {self.transition_points}",
            "  Per-regime details:",
        ]
        for i, seg in enumerate(self.segments):
            lines.append(
                f"    Regime {i}: [{seg.start_idx}-{seg.end_idx}] "
                f"τ={seg.tau} m={seg.m} r={seg.r} "
                f"max|f|={seg.max_forcing:.3f} "
                f"edge={seg.edge_score:.3f} "
                f"LLE={seg.lyapunov:+.4f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


class AdaptiveHAVOK:
    """Non-stationary HAVOK with automatic regime detection and adaptation.

    The algorithm works in three phases:
    1. Detect regime shifts in the raw signal (statistical + spectral)
    2. For each regime segment, auto-tune optimal HAVOK parameters
    3. Run HAVOK on each segment with its own parameters, stitch results

    Parameters
    ----------
    detection_method : str
        'changepoint' — Bayesian changepoint detection (default, robust)
        'spectral' — spectral density change detection
        'rolling' — rolling statistics divergence
    min_segment_length : int
        Minimum samples per regime (avoids over-fragmentation)
    max_segments : int
        Maximum number of regimes to detect
    significance : float
        Changepoint significance level (0.01-0.10)
    """

    def __init__(
        self,
        detection_method: str = "changepoint",
        min_segment_length: int = 100,
        max_segments: int = 10,
        significance: float = 0.05,
        base_tau: int = 1,
        base_m: int = 50,
        base_r: int = 5,
    ):
        self.detection_method = detection_method
        self.min_segment_length = min_segment_length
        self.max_segments = max_segments
        self.significance = significance
        self.base_tau = base_tau
        self.base_m = base_m
        self.base_r = base_r

    def fit_transform(
        self,
        X: np.ndarray,
        t: Optional[np.ndarray] = None,
        show_progress: bool = True,
    ) -> AdaptiveResult:
        """Detect regimes and run adaptive HAVOK.

        Args:
            X: univariate time series
            t: optional time values
            show_progress: show progress bar

        Returns:
            AdaptiveResult with per-segment forcing and transition points
        """
        X = np.asarray(X).ravel()
        n = len(X)
        if t is None:
            t = np.arange(n, dtype=float)

        # Phase 1: Detect regime shifts
        transition_points = self._detect_regimes(X)
        logger.info(f"Detected {len(transition_points)} regime transitions")

        # Add boundaries
        all_points = [0] + sorted(transition_points) + [n]
        # Filter segments that are too short
        segments_raw = []
        for i in range(len(all_points) - 1):
            start, end = all_points[i], all_points[i + 1]
            if end - start >= self.min_segment_length:
                segments_raw.append((start, end))

        # Merge adjacent tiny segments
        segments = self._merge_small_segments(segments_raw)

        if show_progress:
            logger.info(f"Analyzing {len(segments)} segments...")

        # Phase 2 + 3: Auto-tune + HAVOK per segment
        regime_segments = []
        full_forcing = np.zeros(n)
        full_risk = np.zeros(n, dtype=int)
        tau_timeline = np.zeros(n, dtype=int)
        m_timeline = np.zeros(n, dtype=int)

        for seg_idx, (start, end) in enumerate(segments):
            x_seg = X[start:end]

            # Auto-tune for this segment
            tau_s, m_s = self._auto_tune_segment(x_seg)

            # Run HAVOK
            from havolib.pipeline import HavokPipeline
            pipe = HavokPipeline(tau=tau_s, m=m_s, r=self.base_r,
                                 threshold_std=3.0, window=min(self.min_segment_length, end - start))
            pipe.fit(np.arange(len(x_seg)), x_seg)

            forcing = pipe.get_forcing()
            risk = pipe.get_risk().astype(int)

            # Edge of chaos
            from havolib.edge_of_chaos import edge_of_chaos_score
            try:
                eoc = edge_of_chaos_score(x_seg, tau=tau_s, m=min(m_s, 50))
            except Exception:
                eoc = {"largest_lyapunov_exponent": 0.0, "edge_of_chaos_score": 0.0, "interpretation": "unknown"}

            # Pad to fit segment boundaries
            pad_before = max(0, len(x_seg) - len(forcing))
            if pad_before > 0:
                forcing = np.concatenate([np.zeros(pad_before), forcing])
                risk = np.concatenate([np.zeros(pad_before, dtype=int), risk])

            seg_len = min(end - start, len(forcing))
            full_forcing[start:start + seg_len] = forcing[:seg_len]
            full_risk[start:start + seg_len] = risk[:seg_len]
            tau_timeline[start:end] = tau_s
            m_timeline[start:end] = m_s

            regime_segments.append(RegimeSegment(
                start_idx=start, end_idx=end,
                tau=tau_s, m=m_s, r=self.base_r,
                forcing=forcing[:seg_len],
                risk=risk[:seg_len],
                max_forcing=float(np.max(np.abs(forcing[:seg_len]))),
                n_risk_events=int(np.sum(risk[:seg_len])),
                edge_score=eoc["edge_of_chaos_score"],
                lyapunov=eoc["largest_lyapunov_exponent"],
            ))

        transition_idx = [s.start_idx for s in regime_segments[1:]] if len(regime_segments) > 1 else []

        return AdaptiveResult(
            n_samples=n,
            segments=regime_segments,
            transition_points=transition_idx,
            full_forcing=full_forcing,
            full_risk=full_risk,
            parameter_timeline={"tau": tau_timeline, "m": m_timeline},
        )

    def _detect_regimes(self, X: np.ndarray) -> List[int]:
        """Detect regime shift points in the raw signal."""
        n = len(X)

        if self.detection_method == "changepoint":
            return self._changepoint_detect(X)
        elif self.detection_method == "spectral":
            return self._spectral_detect(X)
        else:
            return self._rolling_detect(X)

    def _changepoint_detect(self, X: np.ndarray) -> List[int]:
        """Bayesian changepoint detection via ruptures."""
        try:
            import ruptures as rpt
            algo = rpt.Pelt(model="rbf", min_size=self.min_segment_length).fit(X)
            result = algo.predict(pen=self.significance * np.log(len(X)))
            # Remove the last point (which is always n)
            return [p for p in result[:-1] if p < len(X) - self.min_segment_length]
        except ImportError:
            logger.warning("ruptures not installed — falling back to rolling detection")
            return self._rolling_detect(X)

    def _spectral_detect(self, X: np.ndarray) -> List[int]:
        """Detect shifts in spectral density."""
        n = len(X)
        points = []
        window = max(50, n // 20)

        for i in range(window, n - window, window):
            prev_spec = np.abs(np.fft.fft(X[i - window:i]))[:window // 2]
            next_spec = np.abs(np.fft.fft(X[i:i + window]))[:window // 2]
            if len(prev_spec) == len(next_spec) and len(prev_spec) > 0:
                # Spectral correlation
                corr = np.corrcoef(prev_spec, next_spec)[0, 1] if np.std(prev_spec) > 1e-10 else 1.0
                if corr < 0.7:  # abrupt spectral change
                    points.append(i)
        return points[:self.max_segments]

    def _rolling_detect(self, X: np.ndarray) -> List[int]:
        """Detect shifts via rolling mean+std divergence."""
        n = len(X)
        window = max(50, n // 20)
        points = []
        scores = np.zeros(n)

        for i in range(window, n - window):
            before = X[i - window:i]
            after = X[i:i + window]
            # Welch's t-test for mean shift
            mean_diff = abs(np.mean(before) - np.mean(after))
            pooled_std = np.sqrt((np.var(before) + np.var(after)) / 2) + 1e-12
            scores[i] = mean_diff / pooled_std

        threshold = np.percentile(scores, 95)
        for i in range(window, n - window, self.min_segment_length):
            if scores[i] > threshold:
                points.append(i)

        return points[:self.max_segments]

    def _auto_tune_segment(self, x: np.ndarray) -> Tuple[int, int]:
        """Auto-tune tau and m for a signal segment."""
        from havolib.auto_tune import optimal_tau_mi, optimal_m_fnn
        try:
            tau = max(1, optimal_tau_mi(x, max_lag=min(100, len(x) // 4)))
            m = max(5, optimal_m_fnn(x, tau, max_m=min(50, len(x) // 10)))
        except Exception:
            tau = self.base_tau
            m = self.base_m
        return tau, m

    def _merge_small_segments(self, segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge segments that are too small."""
        if len(segments) <= 1:
            return segments

        merged = []
        current_start, current_end = segments[0]
        for start, end in segments[1:]:
            if current_end - current_start < self.min_segment_length:
                # Merge with next
                current_end = end
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))
        return merged[:self.max_segments]
