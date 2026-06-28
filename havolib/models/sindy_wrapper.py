"""SINDy model wrapper — sparse identification of nonlinear dynamics.

Wraps the `pysindy` library.  Only registered if `pysindy` is importable
— otherwise users get a clear error message when requesting model_type="sindy".

Install:  pip install havok-toolbox[sindy]
"""
import numpy as np
from typing import Optional
from .base import BaseRegimeModel
from .registry import ModelRegistry
from ._utils import is_available

if is_available("pysindy"):
    import pysindy as ps

    @ModelRegistry.register("sindy")
    class SINDyWrapper(BaseRegimeModel):
        """Regime detection via SINDy model-data mismatch.

        Fits a sparse dynamical model (SINDy) to the data.  The forcing
        signal is the pointwise L2 norm of the prediction error between
        the SINDy derivative and the finite-difference derivative.
        """

        def __init__(self, **kwargs):
            self._model = ps.SINDy(**kwargs)
            self._forcing: Optional[np.ndarray] = None
            self._risk: Optional[np.ndarray] = None

        def fit(self, t: Optional[np.ndarray], x: np.ndarray) -> "SINDyWrapper":
            x_arr = np.asarray(x, dtype=float).reshape(-1, 1)
            self._model.fit(x_arr, t=t)
            return self

        def transform(self, t: Optional[np.ndarray], x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=float).reshape(-1, 1)
            x_dot_pred = self._model.predict(x_arr)
            if t is not None:
                x_dot_real = np.gradient(x_arr.ravel(), t, edge_order=2)
            else:
                x_dot_real = np.gradient(x_arr.ravel())
            self._forcing = np.linalg.norm(
                x_dot_real.reshape(-1, 1) - x_dot_pred, axis=1
            )
            return self._forcing

        def get_risk(self) -> np.ndarray:
            if self._forcing is None:
                raise RuntimeError("Call fit() or transform() first.")
            f = self._forcing
            self._risk = (f - f.min()) / (f.max() - f.min() + 1e-12)
            return self._risk
else:
    # pysindy not installed — the "sindy" key will simply not be in the registry.
    # ModelRegistry.get("sindy") will raise a clear ValueError.
    pass
