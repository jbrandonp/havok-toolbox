"""
User-facing analysis tools — what real users actually need.

- Confidence intervals via bootstrap
- Export to CSV/JSON/MATLAB
- Progress bars (tqdm)
- Multi-channel batch processing
- Parameter recommendation with explanations
- AnalysisReport: summary metrics

Usage:
    from havolib.user import analyze, batch_analyze, suggest_and_explain
    report = analyze(my_data)  # returns AnalysisReport
    report.export("results.csv")
    print(report.summary())
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import json
import time
import warnings

from havolib.pipeline import HavokPipeline
from havolib.edge_of_chaos import edge_of_chaos_score


# ── Progress bar (optional tqdm) ──────────────────────────────

def _get_progress():
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        return lambda x, **kw: x  # no-op


# ── Confidence intervals via bootstrap ────────────────────────

def bootstrap_forcing(
    x: np.ndarray,
    n_bootstrap: int = 100,
    tau: int = 1,
    m: int = 50,
    r: int = 5,
    confidence: float = 0.95,
    seed: int = 42,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute bootstrap confidence intervals for the forcing signal.

    Resamples the time series with block bootstrap (preserves autocorrelation)
    and recomputes HAVOK on each resample.

    Args:
        x: time series
        n_bootstrap: number of bootstrap iterations (100+ recommended)
        tau, m, r: HAVOK parameters
        confidence: confidence level (0.95 = 95% CI)

    Returns:
        dict with keys: forcing_mean, forcing_lower, forcing_upper,
                        risk_probability (continuous 0-1), significant_mask
    """
    n = len(x)
    block_size = max(10, int(np.sqrt(n)))  # block bootstrap preserves autocorrelation
    n_blocks = n // block_size

    rng = np.random.default_rng(seed)
    forcings = []

    iterator = range(n_bootstrap)
    if show_progress:
        tqdm = _get_progress()
        iterator = tqdm(iterator, desc="Bootstrap", unit="iter")

    try:
        for _ in iterator:
            # Block bootstrap resample
            idx = rng.integers(0, n_blocks, size=n_blocks)
            x_boot = np.concatenate([
                x[i * block_size:(i + 1) * block_size]
                for i in idx if i * block_size + block_size <= n
            ])

            if len(x_boot) < m * tau * 2:
                continue

            try:
                t_boot = np.arange(len(x_boot))
                pipe = HavokPipeline(tau=tau, m=m, r=r)
                pipe.fit(t_boot, x_boot)
                f = pipe.get_forcing()

                # Pad to match original length
                if len(f) < n:
                    pad = np.zeros(n - len(f))
                    f = np.concatenate([pad, f])
                forcings.append(f[:n])
            except Exception:
                continue
    except KeyboardInterrupt:
        warnings.warn("Bootstrap interrupted — using available samples")

    if len(forcings) < 10:
        warnings.warn(f"Only {len(forcings)} bootstrap samples — results unreliable")
        if not forcings:
            return {"forcing_mean": np.zeros(n), "forcing_lower": np.zeros(n),
                    "forcing_upper": np.zeros(n), "risk_probability": np.zeros(n),
                    "significant_mask": np.zeros(n, dtype=bool)}

    F = np.array(forcings)
    alpha = (1 - confidence) / 2

    forcing_mean = np.mean(F, axis=0)
    forcing_lower = np.percentile(F, alpha * 100, axis=0)
    forcing_upper = np.percentile(F, (1 - alpha) * 100, axis=0)

    # Risk probability: fraction of bootstrap samples where |forcing| > threshold
    threshold = 2.0 * np.std(forcing_mean) + np.mean(np.abs(forcing_mean))
    risk_prob = np.mean(np.abs(F) > threshold, axis=0)

    # Statistically significant regions (CI doesn't contain zero)
    significant = np.sign(forcing_lower) == np.sign(forcing_upper)

    return {
        "forcing_mean": forcing_mean,
        "forcing_lower": forcing_lower,
        "forcing_upper": forcing_upper,
        "risk_probability": risk_prob,
        "significant_mask": significant,
    }


# ── Export ─────────────────────────────────────────────────────

def export_csv(path: str, time: np.ndarray, forcing: np.ndarray, risk: np.ndarray,
               extra: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Export HAVOK results to CSV."""
    df = pd.DataFrame({"time": time, "forcing": forcing, "risk": risk.astype(int)})
    if extra:
        for name, arr in extra.items():
            if len(arr) == len(time):
                df[name] = arr
    df.to_csv(path, index=False)


def export_json(path: str, metrics: Dict[str, Any]) -> None:
    """Export HAVOK metrics to JSON."""
    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            serializable[k] = float(v)
        elif isinstance(v, dict):
            serializable[k] = {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv
                               for kk, vv in v.items()}
        else:
            serializable[k] = v
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)


# ── Analysis Report ────────────────────────────────────────────

@dataclass
class AnalysisReport:
    """Comprehensive HAVOK analysis report with user-friendly metrics."""

    # Input
    data_label: str = ""
    n_samples: int = 0

    # HAVOK parameters
    tau: int = 0
    m: int = 0
    r: int = 0

    # Forcing metrics
    max_forcing: float = 0.0
    mean_abs_forcing: float = 0.0
    forcing_std: float = 0.0

    # Risk metrics
    n_risk_events: int = 0
    risk_fraction: float = 0.0
    max_risk_duration: int = 0

    # Edge of chaos
    lyapunov_exponent: float = 0.0
    edge_score: float = 0.0
    edge_interpretation: str = ""

    # Confidence
    confidence_level: float = 0.0
    significant_fraction: float = 0.0

    # Arrays (not printed in summary)
    forcing: Optional[np.ndarray] = field(default=None, repr=False)
    risk: Optional[np.ndarray] = field(default=None, repr=False)
    eigen_coords: Optional[np.ndarray] = field(default=None, repr=False)
    bootstrap: Optional[Dict] = field(default=None, repr=False)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            f"HAVOK ANALYSIS REPORT — {self.data_label}",
            "=" * 60,
            f"  Samples: {self.n_samples:,}  |  Parameters: τ={self.tau}, m={self.m}, r={self.r}",
            "",
            "  FORCING SIGNAL",
            f"    Max |forcing|: {self.max_forcing:.4f}",
            f"    Mean |forcing|: {self.mean_abs_forcing:.4f}",
            f"    Std forcing:   {self.forcing_std:.4f}",
            "",
            "  REGIME-SHIFT RISK",
            f"    Risk events: {self.n_risk_events} ({self.risk_fraction:.1%} of data)",
            f"    Max duration: {self.max_risk_duration} samples",
            "",
            "  EDGE OF CHAOS",
            f"    Lyapunov exponent: {self.lyapunov_exponent:+.4f}",
            f"    Edge score: {self.edge_score:.3f}",
            f"    → {self.edge_interpretation}",
            "",
            "  CONFIDENCE",
            f"    Significant regions: {self.significant_fraction:.1%}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def export(self, path: str) -> None:
        """Export to CSV (or JSON if .json extension)."""
        path = Path(path)
        if path.suffix == ".json":
            export_json(str(path), {
                "max_forcing": self.max_forcing,
                "n_risk_events": self.n_risk_events,
                "lyapunov_exponent": self.lyapunov_exponent,
                "edge_score": self.edge_score,
            })
        else:
            if self.forcing is not None:
                t = np.arange(len(self.forcing))
                r = self.risk if self.risk is not None else np.zeros(len(self.forcing))
                export_csv(str(path), t, self.forcing, r)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith("_") and not isinstance(v, np.ndarray)}


# ── Main analysis function ─────────────────────────────────────

def analyze(
    x: np.ndarray,
    label: str = "data",
    tau: int = 1,
    m: int = 50,
    r: int = 5,
    threshold_std: float = 3.0,
    window: int = 100,
    bootstrap_ci: bool = False,
    n_bootstrap: int = 100,
    show_progress: bool = True,
) -> AnalysisReport:
    """Run complete HAVOK analysis with user-friendly output.

    Args:
        x: univariate time series
        label: name for the report
        tau, m, r: HAVOK parameters
        bootstrap_ci: compute bootstrap confidence intervals
        n_bootstrap: bootstrap iterations (if enabled)

    Returns:
        AnalysisReport with summary metrics, arrays, and export methods
    """
    t = np.arange(len(x))
    report = AnalysisReport(data_label=label, n_samples=len(x), tau=tau, m=m, r=r)

    # HAVOK pipeline
    if show_progress:
        print(f"Running HAVOK on '{label}' ({len(x):,} samples)...", end=" ")

    t0 = time.perf_counter()
    pipe = HavokPipeline(tau=tau, m=m, r=r, threshold_std=threshold_std, window=window)
    pipe.fit(t, x)
    elapsed = time.perf_counter() - t0

    if show_progress:
        print(f"done in {elapsed:.2f}s")

    forcing = pipe.get_forcing()
    risk = pipe.get_risk()
    V = pipe.get_eigen_coordinates()

    report.forcing = forcing
    report.risk = risk
    report.eigen_coords = V

    # Forcing metrics
    abs_f = np.abs(forcing)
    report.max_forcing = float(np.max(abs_f))
    report.mean_abs_forcing = float(np.mean(abs_f))
    report.forcing_std = float(np.std(abs_f))

    # Risk metrics
    report.n_risk_events = int(np.sum(risk))
    report.risk_fraction = float(np.mean(risk))

    # Longest consecutive risk event
    if report.n_risk_events > 0:
        diffs = np.diff(np.where(risk)[0])
        max_dur = 1
        cur_dur = 1
        for d in diffs:
            if d == 1:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 1
        report.max_risk_duration = max_dur

    # Edge of chaos
    eoc = edge_of_chaos_score(x, tau=tau, m=min(m, 50))
    report.lyapunov_exponent = eoc["largest_lyapunov_exponent"]
    report.edge_score = eoc["edge_of_chaos_score"]
    report.edge_interpretation = eoc["interpretation"]

    # Bootstrap confidence intervals
    if bootstrap_ci:
        report.bootstrap = bootstrap_forcing(
            x, n_bootstrap=n_bootstrap, tau=tau, m=m, r=r,
            show_progress=show_progress,
        )
        report.confidence_level = 0.95
        report.significant_fraction = float(np.mean(report.bootstrap["significant_mask"]))

    return report


# ── Batch processing ───────────────────────────────────────────

def batch_analyze(
    files: List[str],
    column: Optional[str] = None,
    tau: int = 1,
    m: int = 50,
    r: int = 5,
    output_dir: Optional[str] = None,
    show_progress: bool = True,
) -> List[AnalysisReport]:
    """Run HAVOK on multiple CSV files.

    Args:
        files: list of CSV file paths
        column: column name (None = auto-detect numeric column)
        tau, m, r: HAVOK parameters
        output_dir: if set, export each report as CSV + JSON

    Returns:
        list of AnalysisReport objects
    """
    reports = []
    iterator = files
    if show_progress:
        tqdm = _get_progress()
        iterator = tqdm(files, desc="Batch HAVOK", unit="file")

    for fpath in iterator:
        try:
            df = pd.read_csv(fpath)
            if column is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    continue
                col = numeric_cols[0]
            else:
                col = column

            x = df[col].values.astype(float)
            label = Path(fpath).stem
            report = analyze(x, label=label, tau=tau, m=m, r=r, show_progress=False)
            reports.append(report)

            if output_dir:
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                report.export(str(out / f"{label}_havok.csv"))
                export_json(str(out / f"{label}_havok.json"), report.to_dict())

        except Exception as e:
            warnings.warn(f"Failed to process {fpath}: {e}")

    return reports


# ── Parameter recommendation with explanation ──────────────────

def suggest_and_explain(
    x: np.ndarray,
    max_lag: int = 100,
    max_m: int = 50,
) -> Dict[str, Any]:
    """Suggest HAVOK parameters with human-readable explanation.

    Returns a dict with tau, m, explanation, and quality assessment.
    """
    from havolib.auto_tune import optimal_tau_mi, optimal_m_fnn, mutual_information

    x = np.asarray(x).ravel()
    x = x - np.mean(x)

    tau = optimal_tau_mi(x, max_lag=max_lag)
    m = optimal_m_fnn(x, tau, max_m=max_m)

    # Quality check
    mi_at_tau = mutual_information(x[:-tau], x[tau:]) if tau < len(x) - 1 else 0
    n = len(x)
    points_per_dim = n / m if m > 0 else 0

    # Recommendations
    if tau <= 2:
        tau_advice = "Low τ suggests the data is fast-sampled or highly autocorrelated. You may want to downsample or increase τ to capture slower dynamics."
    elif tau > max_lag * 0.5:
        tau_advice = f"High τ ({tau}) — first MI minimum is far out. Your data may have long-range correlations. Consider if this is physically meaningful."
    else:
        tau_advice = f"τ={tau} is a reasonable trade-off between information content and redundancy."

    if m < 10:
        m_advice = f"Low m ({m}) — your data's dynamics appear low-dimensional. This is efficient but may miss subtle structure. Try m=15-20 to verify."
    elif m > 40:
        m_advice = f"High m ({m}) — your data's dynamics are high-dimensional. Ensure you have enough samples ({n}) to support this embedding (rule: n > 10×m×τ)."
    else:
        m_advice = f"m={m} captures the dominant dynamics well for your data."

    quality = "good" if (3 < tau < max_lag * 0.4 and 10 < m < 40 and points_per_dim > 20) else "fair"

    return {
        "tau": tau,
        "m": m,
        "method": "Mutual Information (τ) + False Nearest Neighbors (m)",
        "explanation": f"{tau_advice}\n{m_advice}",
        "quality": quality,
        "points_per_dimension": int(points_per_dim),
        "warning": "Low sample-to-dimension ratio — consider shorter m or more data"
                   if points_per_dim < 15 else None,
    }
