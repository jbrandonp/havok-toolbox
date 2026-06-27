"""
AutoML for HAVOK — automatic hyperparameter optimization via Optuna.

Finds optimal (tau, m, r, threshold_std, diff_method) for any dataset.

Usage:
    from havolib.automl import auto_optimize
    best_params = auto_optimize(my_data, n_trials=100)
    # Then use best_params with HavokEstimator(**best_params)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, List, Callable
import logging

logger = logging.getLogger("havok.automl")

_OPTUNA_AVAILABLE = False
try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    pass


def _havok_objective(
    trial: "optuna.Trial",
    X: np.ndarray,
    scoring: str,
    min_samples: int,
) -> float:
    """Optuna objective function for HAVOK hyperparameter search."""
    from havolib.estimator import HavokEstimator

    # Define search space
    tau = trial.suggest_int("tau", 1, min(30, len(X) // 10))
    m = trial.suggest_int("m", 10, min(100, len(X) // 3))
    r = trial.suggest_int("r", 2, min(10, m - 1))
    threshold_std = trial.suggest_float("threshold_std", 1.5, 5.0)
    window = trial.suggest_int("window", 20, min(300, len(X) // 2))
    diff_method = trial.suggest_categorical(
        "diff_method", ["finite_diff", "spline", "gradient", "total_variation"]
    )

    # Ensure r < m
    if r >= m:
        return -1e9

    try:
        est = HavokEstimator(
            tau=tau, m=m, r=r,
            threshold_std=threshold_std,
            window=window,
            diff_method=diff_method,
        )
        est.fit(X)

        if scoring == "max_forcing":
            score = float(np.max(np.abs(est.forcing_)))
        elif scoring == "snr":
            score = est.score(X)
        elif scoring == "combined":
            # High forcing + high edge-of-chaos score
            from havolib.edge_of_chaos import edge_of_chaos_score
            eoc = edge_of_chaos_score(X, tau=tau, m=min(m, 50))
            score = float(np.max(np.abs(est.forcing_))) * 0.5 + eoc["edge_of_chaos_score"] * 0.5
        else:
            score = est.score(X)

        # Penalize trivial results (all zero risk or all one risk)
        risk_frac = float(np.mean(est.risk_))
        if risk_frac < 0.01 or risk_frac > 0.99:
            score *= 0.5

        return score

    except Exception as e:
        logger.debug(f"Trial failed: {e}")
        return -1e9


def auto_optimize(
    X: np.ndarray,
    n_trials: int = 100,
    scoring: str = "combined",
    timeout: Optional[int] = None,
    random_state: int = 42,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Auto-optimize HAVOK hyperparameters.

    Args:
        X: univariate time series
        n_trials: number of Optuna trials (100-500 recommended)
        scoring: 'max_forcing', 'snr', or 'combined' (forcing + edge-of-chaos)
        timeout: max seconds (overrides n_trials if set)
        random_state: seed for reproducibility
        show_progress: show Optuna progress bar

    Returns:
        dict with: best_params, best_score, study (Optuna Study object),
                   top_trials (list of dicts)

    Raises:
        ImportError: if optuna is not installed (pip install optuna)
    """
    if not _OPTUNA_AVAILABLE:
        raise ImportError(
            "optuna is required for AutoML. Install with: pip install optuna"
        )

    X = np.asarray(X).ravel()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    objective_fn = lambda trial: _havok_objective(
        trial, X, scoring, len(X)
    )

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress,
    )

    # Collect top trials
    top_trials = []
    for trial in study.trials[:10]:
        if trial.value is not None and trial.value > -1e8:
            top_trials.append({
                "params": trial.params,
                "score": trial.value,
                "number": trial.number,
            })

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "study": study,
        "top_trials": top_trials,
        "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
    }


def suggest_from_automl(X: np.ndarray, n_trials: int = 50) -> str:
    """Run AutoML and return human-readable recommendation."""
    result = auto_optimize(X, n_trials=n_trials, show_progress=False)
    bp = result["best_params"]

    lines = [
        "🔍 HAVOK AutoML Recommendation",
        f"   Best score: {result['best_score']:.3f} (from {result['n_completed']} trials)",
        f"   τ = {bp['tau']}  |  m = {bp['m']}  |  r = {bp['r']}",
        f"   threshold_std = {bp['threshold_std']:.1f}  |  window = {bp['window']}",
        f"   diff_method = {bp['diff_method']}",
        "",
        "Usage:",
        f"   est = HavokEstimator(tau={bp['tau']}, m={bp['m']}, r={bp['r']}, ...)",
    ]
    return "\n".join(lines)
