import numpy as np
from .embedding import hankel_matrix
from .decomposition import eigen_time_delay
from .forcing import extract_forcing
from .detection import threshold_risk
from .auto_tune import suggest_parameters
from .config import (
    DEFAULT_TAU, DEFAULT_M, DEFAULT_R,
    DEFAULT_THRESHOLD_STD, DEFAULT_WINDOW
)
from .surrogate import (
    surrogate_forcing_distribution,
    validate_forcing_significance
)
from .pre_processing import preprocess


class HavokPipeline:
    """End-to-end HAVOK analysis pipeline with pre-processing + surrogate validation support."""

    def __init__(self, tau=DEFAULT_TAU, m=DEFAULT_M, r=DEFAULT_R,
                 threshold_std=DEFAULT_THRESHOLD_STD, window=DEFAULT_WINDOW,
                 # New pre-processing options (deeper layer)
                 do_preprocess: bool = False,
                 interpolate: bool = True,
                 smooth_method: str = 'savgol',
                 smooth_window: int = 11,
                 outlier_method: str = 'iqr',
                 detrend: bool = False):
        self.tau = tau
        self.m = m
        self.r = r
        self.threshold_std = threshold_std
        self.window = window

        # Pre-processing config
        self.do_preprocess = do_preprocess
        self.interpolate = interpolate
        self.smooth_method = smooth_method
        self.smooth_window = smooth_window
        self.outlier_method = outlier_method
        self.detrend = detrend

        self.V_ = None
        self.forcing_ = None
        self.risk_ = None
        self.t_ = None
        self.x_ = None
        self.surrogate_maxes_ = None
        self.surrogate_threshold_ = None
        self.is_significant_ = None
        self.p_value_ = None
        self.x_raw_ = None   # original before preprocessing

    def fit(self, t: np.ndarray, x: np.ndarray) -> 'HavokPipeline':
        """Run full HAVOK pipeline on (t, x)."""
        x = np.asarray(x, dtype=float)
        if t is None:
            t = np.arange(len(x))
        t = np.asarray(t, dtype=float)
        if len(t) != len(x):
            raise ValueError("t and x must have the same length.")

        self.x_raw_ = np.asarray(x, dtype=float).copy()

        # Apply pre-processing if enabled (deeper layer best practice)
        if self.do_preprocess:
            x = preprocess(
                x,
                interpolate=self.interpolate,
                smooth_method=self.smooth_method,
                smooth_window=self.smooth_window,
                outlier_method=self.outlier_method,
                detrend=self.detrend
            )
            print("[HAVOK] Pre-processing applied (interpolate/outlier/smooth).")

        # Basic length sanity check (per deeper layer guidance)
        if len(x) < 10 * self.m:
            print(f"[HAVOK WARNING] Data length {len(x)} may be too short for m={self.m} (recommend >10*m).")

        H = hankel_matrix(x, self.m, self.tau)
        t_hankel = t[:H.shape[0]]

        V, _ = eigen_time_delay(H, self.r)
        forcing = extract_forcing(V, t_hankel)
        risk = threshold_risk(forcing, self.window, self.threshold_std)

        self.V_ = V
        self.forcing_ = forcing
        self.risk_ = risk
        self.t_ = t_hankel
        self.x_ = x[:H.shape[0]]
        return self

    def get_forcing(self) -> np.ndarray:
        if self.forcing_ is None:
            raise RuntimeError("Call fit() first.")
        return self.forcing_

    def get_risk(self) -> np.ndarray:
        if self.risk_ is None:
            raise RuntimeError("Call fit() first.")
        return self.risk_

    def get_eigen_coordinates(self) -> np.ndarray:
        if self.V_ is None:
            raise RuntimeError("Call fit() first.")
        return self.V_

    def suggest_parameters(self, data: np.ndarray, max_lag: int = 100, max_m: int = 50) -> dict:
        """
        Use Mutual Information + practical m to recommend good tau and m.
        """
        return suggest_parameters(data, max_lag=max_lag, max_m=max_m)

    def auto_fit(self, t: np.ndarray, x: np.ndarray, max_lag: int = 100, max_m: int = 50,
                 do_preprocess: bool = None) -> 'HavokPipeline':
        """Automatically choose tau and m using MI, then fit."""
        if do_preprocess is not None:
            self.do_preprocess = do_preprocess

        x = np.asarray(x, dtype=float)
        if t is None:
            t = np.arange(len(x))
        t = np.asarray(t, dtype=float)

        # Pre-process *before* parameter suggestion when enabled
        x_for_params = x
        if self.do_preprocess:
            x_for_params = preprocess(
                x,
                interpolate=self.interpolate,
                smooth_method=self.smooth_method,
                smooth_window=self.smooth_window,
                outlier_method=self.outlier_method,
                detrend=self.detrend
            )

        params = self.suggest_parameters(x_for_params, max_lag=max_lag, max_m=max_m)
        self.tau = params["tau"]
        self.m = params["m"]
        print(f"[HAVOK] Auto-selected tau={self.tau}, m={self.m} via Mutual Information")
        return self.fit(t, x)

    def validate_with_surrogates(self, n_surrogates: int = 100, alpha: float = 0.01, seed: int = 42) -> dict:
        """
        Run phase-randomized (Fourier) surrogate test (deeper layer statistical validation).

        This is the key to knowing whether forcing spikes are real or just autocorrelation.
        """
        if self.forcing_ is None or self.x_ is None:
            raise RuntimeError("Call fit() or auto_fit() first before surrogate validation.")

        observed_max = float(np.max(np.abs(self.forcing_)))

        def make_pipe():
            # Replicate current hyperparameters + preproc settings
            return HavokPipeline(
                tau=self.tau, m=self.m, r=self.r,
                threshold_std=self.threshold_std, window=self.window,
                do_preprocess=self.do_preprocess,
                interpolate=self.interpolate,
                smooth_method=self.smooth_method,
                smooth_window=self.smooth_window,
                outlier_method=self.outlier_method,
                detrend=self.detrend
            )

        print(f"[HAVOK] Running {n_surrogates} phase-randomized surrogates for statistical validation...")
        surr_maxes, thresh = surrogate_forcing_distribution(
            self.x_, make_pipe, n_surrogates=n_surrogates, seed=seed
        )

        self.surrogate_maxes_ = surr_maxes
        self.surrogate_threshold_ = thresh
        self.is_significant_, self.p_value_ = validate_forcing_significance(observed_max, surr_maxes, alpha=alpha)

        result = {
            "observed_max_forcing": observed_max,
            "surrogate_99th_percentile": thresh,
            "n_surrogates": len(surr_maxes),
            "p_value": self.p_value_,
            "significant_at_alpha": self.is_significant_,
            "alpha": alpha,
            "recommendation": "Significant intermittent forcing detected (likely real regime-shift driver)."
                          if self.is_significant_ else "Forcing not statistically significant vs linear surrogates."
        }
        print(f"[HAVOK] Surrogate validation: p={self.p_value_:.3f}, 99% thresh={thresh:.4f}, significant={self.is_significant_}")
        return result

    def get_surrogate_threshold(self) -> float:
        if self.surrogate_threshold_ is None:
            raise RuntimeError("Call validate_with_surrogates() first.")
        return self.surrogate_threshold_
