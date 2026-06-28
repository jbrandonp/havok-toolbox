import numpy as np
import logging
from typing import Optional
from .embedding import hankel_matrix
from .decomposition import eigen_time_delay
from .forcing import extract_forcing
from .detection import threshold_risk

logger = logging.getLogger("havok.pipeline")
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
    """End-to-end HAVOK analysis pipeline with pre-processing + surrogate validation support.

    Parameters
    ----------
    tau : int
        Time delay for embedding.
    m : int
        Embedding dimension (Hankel columns).
    r : int
        Number of eigen-time-delay coordinates (SVD rank).
    threshold_std : float
        Risk threshold in std devs.
    window : int
        Rolling window for risk computation.
    do_preprocess : bool
        Whether to apply interpolation/outlier removal/smoothing.
    interpolate : bool
        Interpolate missing values.
    smooth_method : {'savgol', 'lowpass', None}
        Smoothing method.
    smooth_window : int
        Smoothing window size.
    outlier_method : {'iqr', 'zscore', None}
        Outlier removal.
    detrend : bool
        Whether to detrend.
    """

    def __init__(self, tau: int = DEFAULT_TAU, m: int = DEFAULT_M, r: int = DEFAULT_R,
                 threshold_std: float = DEFAULT_THRESHOLD_STD, window: int = DEFAULT_WINDOW,
                 # New pre-processing options (deeper layer)
                 do_preprocess: bool = False,
                 interpolate: bool = True,
                 smooth_method: str = 'savgol',
                 smooth_window: int = 11,
                 outlier_method: str = 'iqr',
                 detrend: bool = False):
        # Hyper-parameter validation (prevents silent failures)
        for name, val in [("tau", tau), ("m", m), ("r", r)]:
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"{name} must be a positive integer, got {val}")
        if not isinstance(threshold_std, (int, float)) or threshold_std <= 0:
            raise ValueError(f"threshold_std must be > 0, got {threshold_std}")
        if not isinstance(window, int) or window <= 0:
            raise ValueError(f"window must be a positive integer, got {window}")
        if r > m:
            import warnings
            warnings.warn(f"r ({r}) > m ({m}); clamping r to {max(2, m-1)}")
            r = max(2, m - 1)

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
        self.n_skip_ = 0

    def _copy_config(self):
        """Return a new HavokPipeline with identical configuration (avoids fragile manual duplication)."""
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

    def fit(self, t: Optional[np.ndarray], x: np.ndarray) -> 'HavokPipeline':
        """Run full HAVOK pipeline on (t, x) — delegates to HavokEstimator internally.

        Parameters
        ----------
        t : np.ndarray or None
            Time vector, shape (n,). If None, uses np.arange(n).
        x : np.ndarray
            Univariate signal, shape (n,).

        Returns
        -------
        self : HavokPipeline
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError(f"Input signal must be 1-D, got shape {x.shape}.")
        if not np.all(np.isfinite(x)):
            raise ValueError("Input contains NaN or Inf values.")
        if t is None:
            t = np.arange(len(x))
        t = np.asarray(t, dtype=float)
        if len(t) != len(x):
            raise ValueError("t and x must have the same length.")
        if not np.all(np.isfinite(t)):
            raise ValueError("t contains NaN or Inf values.")

        self.x_raw_ = np.asarray(x, dtype=float).copy()

        # Apply pre-processing if enabled
        if self.do_preprocess:
            x = preprocess(
                x,
                interpolate=self.interpolate,
                smooth_method=self.smooth_method,
                smooth_window=self.smooth_window,
                outlier_method=self.outlier_method,
                detrend=self.detrend
            )
            logger.info("Pre-processing applied (interpolate/outlier/smooth).")

        # Basic length sanity check
        if len(x) < 10 * self.m:
            logger.warning(f"Data length {len(x)} may be too short for m={self.m} (recommend >10*m).")

        # Delegate to HavokEstimator (single center of truth for HAVOK math)
        from havolib.estimator import HavokEstimator
        est = HavokEstimator(
            tau=self.tau, m=self.m, r=self.r,
            threshold_std=self.threshold_std, window=self.window,
        )
        est.fit(x, t=t)

        # Copy results into pipeline's attribute namespace
        self.n_skip_ = est.n_skip_
        self.V_ = est.eigen_coords_
        self.forcing_ = est.forcing_
        self.risk_ = est.risk_
        self.t_ = t[self.n_skip_:]
        self.x_ = x[self.n_skip_:]
        return self

    def get_forcing(self) -> np.ndarray:
        if self.forcing_ is None:
            raise RuntimeError("Call fit() first.")
        return self.forcing_.copy()  # prevent accidental mutation

    def get_risk(self) -> np.ndarray:
        if self.risk_ is None:
            raise RuntimeError("Call fit() first.")
        return self.risk_.copy()

    def get_eigen_coordinates(self) -> np.ndarray:
        if self.V_ is None:
            raise RuntimeError("Call fit() first.")
        return self.V_.copy()

    def get_n_skip(self) -> int:
        """Number of leading samples trimmed by Hankel (m-1)*tau. Outputs have length n - n_skip_."""
        return getattr(self, 'n_skip_', 0)

    def suggest_parameters(self, data: np.ndarray, max_lag: int = 100, max_m: int = 50) -> dict:
        """
        Use Mutual Information + practical m to recommend good tau and m.
        """
        return suggest_parameters(data, max_lag=max_lag, max_m=max_m)

    def auto_fit(self, t: Optional[np.ndarray], x: np.ndarray, max_lag: int = 100, max_m: int = 50,
                 do_preprocess: bool = None) -> 'HavokPipeline':
        """Automatically choose tau and m using MI, then fit.

        Parameters
        ----------
        t : np.ndarray or None
            Time vector.
        x : np.ndarray
            Signal, shape (n,).
        """
        if do_preprocess is not None:
            self.do_preprocess = do_preprocess

        raw_x = np.asarray(x, dtype=float).copy()
        if raw_x.ndim != 1 or not np.all(np.isfinite(raw_x)):
            raise ValueError("x must be 1-D and finite.")
        if t is None:
            t = np.arange(len(raw_x))
        t = np.asarray(t, dtype=float)

        self.x_raw_ = raw_x.copy()

        # Pre-process *before* parameter suggestion when enabled
        x_for_params = raw_x
        if self.do_preprocess:
            x_for_params = preprocess(
                raw_x,
                interpolate=self.interpolate,
                smooth_method=self.smooth_method,
                smooth_window=self.smooth_window,
                outlier_method=self.outlier_method,
                detrend=self.detrend
            )

        params = self.suggest_parameters(x_for_params, max_lag=max_lag, max_m=max_m)
        self.tau = params["tau"]
        self.m = params["m"]
        self.r = min(self.r, max(2, self.m - 1))
        logger.info(f"Auto-selected tau={self.tau}, m={self.m}, r_clamped={self.r} via Mutual Information")

        # Temporarily disable preprocess flag so fit does not re-process
        orig_do = self.do_preprocess
        self.do_preprocess = False
        result = self.fit(t, x_for_params)
        self.do_preprocess = orig_do
        # Ensure raw is preserved (fit always sets x_raw_ to its input)
        self.x_raw_ = raw_x
        return result

    def validate_with_surrogates(self, n_surrogates: int = 100, alpha: float = 0.01, seed: int = 42) -> dict:
        """
        Run phase-randomized (Fourier) surrogate test (deeper layer statistical validation).

        This is the key to knowing whether forcing spikes are real or just autocorrelation.

        Parameters
        ----------
        n_surrogates : int
            Number of surrogates.
        alpha : float
            Significance level.
        seed : int
            RNG seed.
        """
        if self.forcing_ is None or self.x_raw_ is None:
            raise RuntimeError("Call fit() or auto_fit() first before surrogate validation.")

        observed_max = float(np.max(np.abs(self.forcing_)))

        def make_pipe():
            # Replicate current hyperparameters + preproc settings
            # Surrogates are generated from raw data; preprocessing will be applied once inside fit()
            return self._copy_config()

        logger.info(f"Running {n_surrogates} phase-randomized surrogates for statistical validation...")
        surr_maxes, thresh = surrogate_forcing_distribution(
            self.x_raw_, make_pipe, n_surrogates=n_surrogates, seed=seed
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
        logger.info(f"Surrogate validation: p={self.p_value_:.3f}, 99% thresh={thresh:.4f}, significant={self.is_significant_}")
        return result

    def get_surrogate_threshold(self) -> float:
        if self.surrogate_threshold_ is None:
            raise RuntimeError("Call validate_with_surrogates() first.")
        return self.surrogate_threshold_
