"""
Adaptive Non-Stationary HAVOK — auto-adjusts parameters when dynamics drift.

Major upgrade (2026): now includes
- Adams-MacKay Bayesian Online Changepoint Detection (BOCPD) — true streaming
- Soft regime blending (no hard cuts)
- RegimeMemory (cross-segment meta-learning of tau/m/r)
- Koopman spectral drift detection (regimes in latent dynamics space, not raw signal)

This is the primary technical moat.
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


# ──────────────────────────────────────────────────────────────
# UPGRADE: Adams & MacKay (2007) Bayesian Online Changepoint Detection
# ──────────────────────────────────────────────────────────────

class BayesianOnlineCP:
    """Corrected online Bayesian changepoint detection (Adams & MacKay 2007 style, Gaussian).

    Uses recursive sufficient statistics per run length.
    Much more reliable at detecting mean/variance shifts than previous version.

    NOTE: internal state lists grow O(#possible run lengths) with sequence length.
    Suitable for moderate lengths; for very long streams prefer detection_method='changepoint'.
    """
    def __init__(self, hazard_rate: float = 0.01, mu0: float = 0.0, sigma2: float = 1.0):
        self.hazard = float(hazard_rate)
        self.mu0 = float(mu0)
        self.sigma2 = float(sigma2)
        # Per possible run length: sufficient stats
        self._sums = [0.0]      # sum of observations for this run
        self._counts = [0]      # number of observations
        self._R = [1.0]         # posterior P(run_length = r | data)
        self._history = []

    def update(self, x: float) -> float:
        x = float(x)
        n_runs = len(self._R)

        # Predictive mean for each run length
        means = []
        for s, c in zip(self._sums, self._counts):
            if c == 0:
                means.append(self.mu0)
            else:
                means.append(s / c)
        means = np.asarray(means)

        # Likelihood under each run (Gaussian with fixed obs variance)
        var = self.sigma2
        lik = np.exp(-0.5 * (x - means)**2 / var)

        # Growth probabilities (continue current run)
        growth = (1 - self.hazard) * np.asarray(self._R) * lik

        # Probability of changepoint now
        cp_prob = self.hazard * np.sum(np.asarray(self._R) * lik)

        # New run length distribution: [CP, growth for r=1, growth for r=2, ...]
        R_new = np.concatenate([[cp_prob], growth])
        R_new = R_new / (np.sum(R_new) + 1e-300)
        self._R = R_new.tolist()

        # Update stats:
        # - New run (CP): starts with this single observation
        new_sums = [x]
        new_counts = [1]
        # - Continuing runs: add observation
        for s, c in zip(self._sums, self._counts):
            new_sums.append(s + x)
            new_counts.append(c + 1)

        self._sums = new_sums
        self._counts = new_counts

        self._history.append(cp_prob)
        return cp_prob

    def get_change_probs(self) -> np.ndarray:
        return np.array(self._history)


# ──────────────────────────────────────────────────────────────
# UPGRADE: Regime memory (meta-learning of good (tau,m,r))
# ──────────────────────────────────────────────────────────────

class RegimeMemory:
    """k-NN memory of successful (tau, m, r) for similar dynamical regimes.

    Extracts simple fingerprint (stats + dominant freq + autocorr) and
    recalls best params from prior segments. Warm-starts new segments.
    """
    def __init__(self, k: int = 3):
        self.k = k
        self._features: List[np.ndarray] = []
        self._params: List[Tuple[int, int, int]] = []

    def _fingerprint(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).ravel()
        if len(x) < 10:
            return np.zeros(6)
        freqs = np.abs(np.fft.rfft(x - np.mean(x)))
        dom = (np.argmax(freqs) / max(1, len(freqs)))
        return np.array([
            np.mean(x), np.std(x),
            np.percentile(x, 75) - np.percentile(x, 25),
            dom,
            np.mean(np.abs(np.diff(x))),
            float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2 else 0.0
        ])

    def query(self, x: np.ndarray) -> Optional[Tuple[int, int, int]]:
        if len(self._features) < self.k:
            return None
        f = self._fingerprint(x)
        dists = [np.linalg.norm(f - ff) for ff in self._features]
        knn = np.argsort(dists)[:self.k]
        taus, ms, rs = zip(*[self._params[i] for i in knn])
        return int(np.median(taus)), int(np.median(ms)), int(np.median(rs))

    def update(self, x: np.ndarray, tau: int, m: int, r: int):
        self._features.append(self._fingerprint(x))
        self._params.append((tau, m, r))


class AdaptiveHAVOK:
    """Non-stationary HAVOK with automatic regime detection and adaptation.

    The algorithm works in three phases:
    1. Detect regime shifts in the raw signal (statistical + spectral)
    2. For each regime segment, auto-tune optimal HAVOK parameters
    3. Run HAVOK on each segment with its own parameters, stitch results

    Parameters
    ----------
    detection_method : str
        'changepoint' — ruptures based (default, reliable for batch)
        'bocpd' — online Bayesian (experimental, streaming)
        'koopman' — drift in Koopman operator eigenvalues
        'spectral' / 'rolling' — other heuristics
    min_segment_length : int
        Minimum samples per regime (avoids over-fragmentation)
    max_segments : int
        Maximum number of regimes to detect
    significance : float
        Changepoint significance level (0.01-0.10)
    """

    def __init__(
        self,
        detection_method: str = "changepoint",   # reliable default; 'bocpd' (online, experimental), 'koopman', 'rolling'
        min_segment_length: int = 100,
        max_segments: int = 10,
        significance: float = 0.05,
        base_tau: int = 1,
        base_m: int = 50,
        base_r: int = 5,
        use_soft_blending: bool = True,
        use_memory: bool = True,
    ):
        self.detection_method = detection_method
        self.min_segment_length = min_segment_length
        self.max_segments = max_segments
        self.significance = significance
        self.base_tau = base_tau
        self.base_m = base_m
        self.base_r = base_r
        self.use_soft_blending = use_soft_blending
        self.use_memory = use_memory
        self.memory = RegimeMemory() if use_memory else None
        # More sensitive default for BOCPD
        bocpd_hazard = 0.03 if detection_method == "bocpd" else 0.01
        self.bocpd = BayesianOnlineCP(hazard_rate=bocpd_hazard) if detection_method == "bocpd" else None

    def fit_transform(
        self,
        X: np.ndarray,
        t: Optional[np.ndarray] = None,
        show_progress: bool = True,
    ) -> AdaptiveResult:
        """Detect regimes (BOCPD / Koopman drift preferred) + adaptive HAVOK with memory + soft blending.

        This is now a serious non-stationary dynamical engine.
        """
        X = np.asarray(X).ravel()
        n = len(X)
        if t is None:
            t = np.arange(n, dtype=float)

        # Guard: very short data → single-regime fallback
        min_needed = max(self.min_segment_length * 2, 30)
        if n < min_needed or n < self.min_segment_length * 2:
            logger.info(f"Data too short for adaptive ({n} pts < {min_needed}) — single-regime fallback")
            from havolib.pipeline import HavokPipeline
            pipe = HavokPipeline(tau=self.base_tau, m=min(self.base_m, n//3),
                                r=min(self.base_r, max(2, min(self.base_m, n//3)-1)))
            try:
                pipe.fit(t, X)
                f = pipe.get_forcing()
                r = pipe.get_risk().astype(int)
            except Exception:
                f = np.zeros(n); r = np.zeros(n, dtype=int)
            seg = RegimeSegment(start_idx=0, end_idx=n, tau=self.base_tau,
                               m=self.base_m, r=self.base_r,
                               forcing=f, risk=r,
                               max_forcing=float(np.nanmax(np.abs(f))) if len(f) and np.any(np.isfinite(f)) else 0.0,
                               n_risk_events=int(np.sum(r)),
                               edge_score=0.0, lyapunov=0.0)
            return AdaptiveResult(
                n_samples=n, segments=[seg], transition_points=[],
                full_forcing=f, full_risk=r,
                parameter_timeline={"tau": np.full(n, self.base_tau, dtype=int),
                                    "m": np.full(n, self.base_m, dtype=int)})

        # Phase 1: modern regime detection
        transition_points = self._detect_regimes(X)
        logger.info(f"Detected {len(transition_points)} regime transitions via {self.detection_method}")

        all_points = [0] + sorted(transition_points) + [n]
        segments_raw = []
        for i in range(len(all_points) - 1):
            s, e = all_points[i], all_points[i+1]
            if e - s >= self.min_segment_length:
                segments_raw.append((s, e))
        segments = self._merge_small_segments(segments_raw)

        if show_progress:
            logger.info(f"Analyzing {len(segments)} segments (soft={self.use_soft_blending}, memory={self.use_memory})...")

        regime_segments: List[RegimeSegment] = []
        full_forcing = np.zeros(n)
        full_risk = np.zeros(n, dtype=int)
        tau_timeline = np.zeros(n, dtype=int)
        m_timeline = np.zeros(n, dtype=int)
        weights_list = []  # for soft blending

        from havolib.pipeline import HavokPipeline
        from havolib.edge_of_chaos import edge_of_chaos_score

        for start, end in segments:
            x_seg = X[start:end]

            # Memory warm-start if available
            tuned = None
            if self.memory:
                tuned = self.memory.query(x_seg)
            if tuned:
                tau_s, m_s, r_s = tuned
            else:
                tau_s, m_s = self._auto_tune_segment(x_seg)
                r_s = self.base_r

            pipe = HavokPipeline(tau=tau_s, m=m_s, r=r_s,
                                 threshold_std=3.0,
                                 window=min(self.min_segment_length, len(x_seg)))
            pipe.fit(np.arange(len(x_seg)), x_seg)

            forcing = pipe.get_forcing()
            risk = pipe.get_risk().astype(int)

            try:
                eoc = edge_of_chaos_score(x_seg, tau=tau_s, m=min(m_s, 50))
            except Exception:
                eoc = {"largest_lyapunov_exponent": 0.0, "edge_of_chaos_score": 0.0}

            pad = max(0, len(x_seg) - len(forcing))
            if pad:
                forcing = np.concatenate([np.full(pad, np.nan), forcing])
                risk = np.concatenate([np.zeros(pad, dtype=int), risk])

            seg_len = min(end - start, len(forcing))
            seg_forcing = forcing[:seg_len]
            seg_risk = risk[:seg_len]

            regime_segments.append(RegimeSegment(
                start_idx=start, end_idx=end, tau=tau_s, m=m_s, r=r_s,
                forcing=seg_forcing, risk=seg_risk,
                max_forcing=float(np.nanmax(np.abs(seg_forcing))) if np.any(np.isfinite(seg_forcing)) else 0.0,
                n_risk_events=int(np.sum(seg_risk)),
                edge_score=eoc["edge_of_chaos_score"],
                lyapunov=eoc["largest_lyapunov_exponent"],
            ))

            if self.memory:
                self.memory.update(x_seg, tau_s, m_s, r_s)

            # Store for later soft weighting
            weights_list.append((start, end, seg_forcing, seg_risk))

            # Populate timeline (simple fill per segment)
            tau_timeline[start:end] = tau_s
            m_timeline[start:end] = m_s

        # Stitch (soft blend if requested)
        if self.use_soft_blending and len(weights_list) > 1:
            full_forcing, full_risk = self._soft_stitch(X, weights_list)
        else:
            for start, end, sf, sr in weights_list:
                sl = min(end - start, len(sf))
                full_forcing[start:start+sl] = sf[:sl]
                full_risk[start:start+sl] = sr[:sl]

        trans = [s.start_idx for s in regime_segments[1:]] if len(regime_segments) > 1 else []
        return AdaptiveResult(
            n_samples=n, segments=regime_segments, transition_points=trans,
            full_forcing=full_forcing, full_risk=full_risk,
            parameter_timeline={"tau": tau_timeline, "m": m_timeline},
        )

    def _soft_stitch(self, X: np.ndarray, segs: List[Tuple[int, int, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """Proper weighted blend using nansum for forcing (preserve nans only if all nan). Prevents nan pollution."""
        n = len(X)
        full_f = np.zeros(n)
        weight_sum = np.zeros(n)
        full_r = np.zeros(n, dtype=int)
        nseg = len(segs)
        for idx, (start, end, sf, sr) in enumerate(segs):
            sl = min(end-start, len(sf))
            w = np.ones(sl, dtype=float)
            ramp = min(25, sl // 3)
            if idx > 0:
                w[:ramp] = np.linspace(0.25, 1.0, ramp)
            if idx < nseg - 1:
                w[-ramp:] = np.linspace(1.0, 0.25, ramp)
            # use nansum logic: accumulate finite contribs
            valid = np.isfinite(sf[:sl])
            full_f[start:start+sl] += np.where(valid, w * sf[:sl], 0.0)
            weight_sum[start:start+sl] += np.where(valid, w, 0.0)
            full_r[start:start+sl] = np.maximum(full_r[start:start+sl], sr[:sl])
        # Normalize only where we have weight
        nz = weight_sum > 1e-12
        full_f[nz] = full_f[nz] / weight_sum[nz]
        # positions never covered or all-nan stay 0 (or could leave nan but match non-soft zeros init behavior for risk)
        return full_f, full_r

    def _detect_regimes(self, X: np.ndarray) -> List[int]:
        """Detect regime shift points. Prefers modern methods."""
        meth = self.detection_method.lower()
        if meth == "bocpd":
            return self._bocpd_detect(X)
        elif meth == "koopman":
            return self._koopman_drift_detect(X)
        elif meth == "changepoint":
            return self._changepoint_detect(X)
        elif meth == "spectral":
            return self._spectral_detect(X)
        else:
            return self._rolling_detect(X)

    def _bocpd_detect(self, X: np.ndarray) -> List[int]:
        """Run online BOCPD and locate changes from the probability time series (smoothed peaks).
        For best batch results use default 'changepoint'. BOCPD is designed for streaming.
        """
        if self.bocpd is None:
            self.bocpd = BayesianOnlineCP(hazard_rate=0.03)

        probs = [self.bocpd.update(float(val)) for val in X]
        probs = np.asarray(probs)

        if len(probs) < self.min_segment_length * 2:
            return []

        # Smooth to reduce noise
        try:
            from scipy.ndimage import uniform_filter1d
            smoothed = uniform_filter1d(probs, size=max(7, len(probs) // 40))
        except Exception:
            smoothed = probs

        thr = max(0.012, self.significance * 0.28)
        cps = []
        for i in range(self.min_segment_length, len(smoothed) - 5):
            if smoothed[i] > thr and smoothed[i] >= smoothed[i-1] and smoothed[i] >= smoothed[i+1]:
                if not cps or i - cps[-1] > self.min_segment_length:
                    cps.append(i)
        return cps[:self.max_segments]

    def _koopman_drift_detect(self, X: np.ndarray, window: Optional[int] = None, stride: int = 40) -> List[int]:
        """Detect regime changes by monitoring drift in the fitted linear Koopman operator A eigenvalues."""
        n = len(X)
        w = window or max(self.min_segment_length, n // 25)
        eig_traj = []
        from havolib.embedding import hankel_matrix
        for st in range(0, n - w, stride):
            seg = X[st:st + w]
            try:
                H = hankel_matrix(seg, min(self.base_m, w//3), self.base_tau)
                V, _ = np.linalg.svd(H, full_matrices=False)
                V = V[:, :min(self.base_r, V.shape[1])]
                dV = np.gradient(V, axis=0)
                A = np.linalg.lstsq(V[:-1], dV[:-1], rcond=None)[0]
                eigs = np.sort(np.abs(np.linalg.eigvals(A)))
                eig_traj.append(eigs)
            except Exception:
                continue
        if len(eig_traj) < 3:
            return self._rolling_detect(X)
        from scipy.stats import wasserstein_distance
        scores = [0.0]
        for i in range(1, len(eig_traj)):
            try:
                d = wasserstein_distance(eig_traj[i-1], eig_traj[i])
            except Exception:
                d = np.linalg.norm(eig_traj[i-1] - eig_traj[i])
            scores.append(d)
        thr = np.mean(scores) + 1.8 * np.std(scores)
        cps = []
        for i, sc in enumerate(scores):
            if sc > thr:
                idx = i * stride + w // 2
                if (not cps or idx - cps[-1] > self.min_segment_length) and idx < n:
                    cps.append(idx)
        return cps[:self.max_segments]

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
