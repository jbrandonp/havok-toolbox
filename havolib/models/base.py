"""
Abstract base class for all regime-shift detection models.

Every model — HAVOK, SINDy, ESN, Mamba, Neural ODE — must implement this
contract.  The pipeline is then completely model-agnostic.

Dimension contract
------------------
``t`` can be ``None`` for regularly-sampled data, otherwise ``(n,)``.
``x`` must be 1D ``(n,)`` or 2D ``(n, d)``.  Multivariate input is reduced
to a scalar forcing per time step via L2 norm or per-model logic, so
``transform()`` always returns a 1D ``(n,)`` array.

Portability
-----------
Models are automatically discovered when their wrapper module is imported
(even conditionally).  No manual registration needed.

To add a new model:
    1. Inherit from BaseRegimeModel
    2. Register with @ModelRegistry.register("model_name")
    3. Implement fit(), transform(), get_forcing()
    4. Optional: override get_risk() for custom scoring
    5. Implement save()/load() for persistence
"""
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Optional


class BaseRegimeModel(ABC):
    """Contract for any regime-shift detection model.

    A model consumes (t, x) and produces a forcing signal that quantifies
    deviation from expected dynamics.  The pipeline then converts forcing
    to risk via :class:`RiskCalculator`.

    Parameters
    ----------
    **kwargs : dict
        Forwarded to the underlying implementation.  Each wrapper validates
        its own parameters — invalid keys raise ``TypeError``.
    """

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._params = kwargs

    # ── core contract ────────────────────────────────────────────

    @abstractmethod
    def fit(self, t: Optional[np.ndarray], x: np.ndarray) -> "BaseRegimeModel":
        """Train the model on a time series.

        Parameters
        ----------
        t : ndarray of shape (n,) or None
            Time vector.  ``None`` means the data are regularly sampled.
        x : ndarray of shape (n,) or (n, d)
            Signal values.  1D for univariate, 2D for multivariate.

        Returns
        -------
        self
        """
        ...

    @abstractmethod
    def transform(self, t: Optional[np.ndarray], x: np.ndarray) -> np.ndarray:
        """Produce a 1D forcing (deviation) signal of length ``n``.

        For multivariate input ``(n, d)`` the wrapper must reduce to a
        scalar per time step (e.g. L2 norm of the per-dimension residuals).

        Parameters
        ----------
        t : ndarray or None
        x : ndarray, shape (n,) or (n, d)

        Returns
        -------
        forcing : ndarray, shape (n,)
            Deviation / forcing signal in native units.
        """
        ...

    def get_forcing(self) -> np.ndarray:
        """Return the forcing from the last ``transform()`` call.

        Raises RuntimeError if ``transform()`` has not been called.
        """
        if not hasattr(self, "_forcing") or self._forcing is None:
            raise RuntimeError("Call transform() before get_forcing().")
        return self._forcing

    # ── risk (can be overridden, else pipeline handles it) ──────

    def get_risk(self) -> np.ndarray:
        """Return risk scores in [0, 1] from the last transform().

        The default normalises forcing to [0, 1] via min-max scaling.
        Override for model-specific scoring (e.g. threshold-based).
        """
        f = self.get_forcing()
        fmin, fmax = f.min(), f.max()
        if fmax - fmin < 1e-12:
            return np.zeros_like(f)
        return (f - fmin) / (fmax - fmin)

    # ── sklearn compat ───────────────────────────────────────────

    def fit_transform(
        self, t: Optional[np.ndarray], x: np.ndarray
    ) -> np.ndarray:
        self.fit(t, x)
        return self.transform(t, x)

    # ── persistence (override in subclasses that support it) ────

    def save(self, path: str) -> None:
        """Save model state to disk.  Default: raise NotImplementedError."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support serialisation."
        )

    @classmethod
    def load(cls, path: str) -> "BaseRegimeModel":
        """Load model state from disk.  Default: raise NotImplementedError."""
        raise NotImplementedError(
            f"{cls.__name__} does not support deserialisation."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._params})"
