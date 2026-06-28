"""HAVOK native model wrapper — backward-compatible with existing pipeline."""
import numpy as np
from typing import Optional
from .base import BaseRegimeModel
from .registry import ModelRegistry


@ModelRegistry.register("havok")
class HavokWrapper(BaseRegimeModel):
    """Thin wrapper around the existing HavokPipeline.

    Valid parameters (matching HavokPipeline):
        tau, m, r, threshold_std, window, diff_method, do_preprocess, ...

    Example
    -------
    >>> model = ModelRegistry.get("havok")(tau=1, m=50, r=5)
    >>> model.fit(None, x_signal)
    >>> forcing = model.transform(None, x_signal)
    >>> risk = model.get_risk()
    """

    # Whitelist of known HavokPipeline parameters
    _KNOWN_PARAMS = frozenset({
        "tau", "m", "r", "threshold_std", "window", "diff_method",
        "solver", "random_state", "do_preprocess", "alpha",
    })

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        unknown = set(kwargs) - self._KNOWN_PARAMS
        if unknown:
            raise TypeError(
                f"HavokWrapper: unknown parameter(s) {unknown}. "
                f"Known: {sorted(self._KNOWN_PARAMS)}"
            )
        from havolib.pipeline import HavokPipeline as _P
        self._pipe = _P(**kwargs)
        self._forcing: Optional[np.ndarray] = None

    def fit(self, t: Optional[np.ndarray], x: np.ndarray) -> "HavokWrapper":
        self.logger.info("Fitting HAVOK on %d points", len(x))
        self._pipe.fit(t, x)
        self._forcing = self._pipe.get_forcing()
        return self

    def transform(self, t: Optional[np.ndarray], x: np.ndarray) -> np.ndarray:
        self._pipe.fit(t, x)
        self._forcing = self._pipe.get_forcing()
        return self._forcing

    def get_risk(self) -> np.ndarray:
        if self._forcing is None:
            raise RuntimeError("Call fit() or transform() before get_risk().")
        # Use HAVOK's native threshold-based risk
        return self._pipe.get_risk().astype(int)
