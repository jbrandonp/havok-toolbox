"""
Risk Engine — multi-dimensional risk scoring for streaming HAVOK.

Combines surge potential, trend strength, burst clustering,
and surrogate significance into a single 0-1 risk score.
"""

import numpy as np
from typing import Tuple
from enum import Enum


class RiskLevel(Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    WARNING = "warning"
    CRITICAL = "critical"


class RiskEngine:
    """Multi-dimensional risk assessor for HAVOK forcing signals.

    Computes a continuous risk score from four dimensions:
      - surge_potential: how fast is forcing growing?
      - trend_strength: is elevated forcing persistent?
      - burst_clustering: are spikes clustered in time?
      - significance: is forcing above noise floor?
    """

    def __init__(
        self,
        surge_weight: float = 0.30,
        trend_weight: float = 0.25,
        cluster_weight: float = 0.20,
        significance_weight: float = 0.25,
        surge_threshold: float = 2.0,
        trend_threshold: float = 1.5,
        cluster_threshold: float = 2.0,
    ):
        total = surge_weight + trend_weight + cluster_weight + significance_weight
        self.w_surge = surge_weight / total
        self.w_trend = trend_weight / total
        self.w_cluster = cluster_weight / total
        self.w_sig = significance_weight / total

        self.surge_threshold = surge_threshold
        self.trend_threshold = trend_threshold
        self.cluster_threshold = cluster_threshold

    def assess(self, forcing_window: np.ndarray) -> Tuple[float, RiskLevel, dict]:
        """Compute risk score from a window of forcing values.

        Args:
            forcing_window: recent forcing values (≥ 50 points recommended)

        Returns:
            risk_score: 0.0 to 1.0
            level: RiskLevel enum
            details: dict with per-dimension scores
        """
        f = np.abs(forcing_window)
        if len(f) < 10:
            return 0.0, RiskLevel.NORMAL, {"surge": 0, "trend": 0, "cluster": 0, "significance": 0}

        surge = self._surge_potential(f)
        trend = self._trend_strength(f)
        cluster = self._burst_clustering(f)
        sig = self._significance(f)

        score = (
            self.w_surge * surge +
            self.w_trend * trend +
            self.w_cluster * cluster +
            self.w_sig * sig
        )

        if score >= 0.8:
            level = RiskLevel.CRITICAL
        elif score >= 0.5:
            level = RiskLevel.WARNING
        elif score >= 0.25:
            level = RiskLevel.ELEVATED
        else:
            level = RiskLevel.NORMAL

        return float(min(1.0, max(0.0, score))), level, {
            "surge": float(surge),
            "trend": float(trend),
            "cluster": float(cluster),
            "significance": float(sig),
        }

    def _surge_potential(self, f: np.ndarray) -> float:
        """How fast is forcing growing? Ratio of recent to overall."""
        if len(f) < 5:
            return 0.0
        recent = np.mean(f[-min(10, len(f)):])
        baseline = np.median(f) + 1e-12
        ratio = recent / baseline
        # Sigmoid mapping: ratio > surge_threshold → high surge
        return float(1.0 / (1.0 + np.exp(-3.0 * (ratio - self.surge_threshold))))

    def _trend_strength(self, f: np.ndarray) -> float:
        """Is forcing persistently elevated? Mann-Kendall-like trend."""
        if len(f) < 20:
            return 0.0
        # Simple: fraction of recent points above long-term median
        median = np.median(f)
        recent_frac = np.mean(f[-min(30, len(f)):] > median * self.trend_threshold)
        return float(min(1.0, recent_frac * 2.0))

    def _burst_clustering(self, f: np.ndarray) -> float:
        """Are forcing spikes clustered in time?"""
        if len(f) < 20:
            return 0.0
        threshold = np.std(f) * self.cluster_threshold + np.mean(f)
        spikes = np.where(f > threshold)[0]
        if len(spikes) < 2:
            return 0.0
        # Measure inter-spike intervals — lower variance = more clustered
        intervals = np.diff(spikes)
        if len(intervals) < 2:
            return float(len(spikes) / len(f))
        cv = np.std(intervals) / (np.mean(intervals) + 1e-12)
        # Low CV = clustered = high score
        return float(max(0.0, 1.0 - cv))

    def _significance(self, f: np.ndarray) -> float:
        """Is forcing amplitude above noise floor?"""
        if len(f) < 10:
            return 0.0
        # Signal-to-noise: max / median absolute deviation
        mad = np.median(np.abs(f - np.median(f))) + 1e-12
        snr = np.max(f) / mad
        # Sigmoid: SNR > 5 → significant
        return float(1.0 / (1.0 + np.exp(-0.8 * (snr - 5.0))))
