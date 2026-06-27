"""
Lightweight Echo State Network (ESN) style risk predictor for HAVOK forcing signals.

Inspired by the excellent MagriLab/Tutorials ESN implementation
(https://github.com/MagriLab/Tutorials/tree/master/esn).

Purpose
-------
After HAVOK extracts the intermittent forcing signal f(t), this module
trains a reservoir to forecast future forcing values or the probability
of an upcoming large spike (regime shift / seizure onset).

This implements the "ML-based risk predictor" layer from the project
architecture.

Key patterns extracted:
- Leaky integrator reservoir
- Sparse random reservoir weights with spectral radius scaling
- Washout + open-loop state collection
- Ridge regression readout
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from typing import Optional, Tuple


class ForcingRiskPredictor:
    """
    Simple ESN for predicting HAVOK forcing and estimating regime-shift risk.

    Usage example:
        predictor = ForcingRiskPredictor(reservoir_size=200, spectral_radius=0.95)
        predictor.train(forcing_train, target_train)
        future_forcing, risk = predictor.predict(forcing_recent, horizon=50)
    """

    def __init__(
        self,
        reservoir_size: int = 200,
        input_dim: int = 1,
        spectral_radius: float = 0.95,
        leak_factor: float = 0.3,
        input_scaling: float = 1.0,
        tikhonov: float = 1e-8,
        connectivity: float = 0.1,
        random_seed: Optional[int] = 42,
    ):
        self.N = reservoir_size
        self.input_dim = input_dim
        self.rho = spectral_radius
        self.alpha = leak_factor
        self.sigma_in = input_scaling
        self.tikh = tikhonov
        self.connectivity = connectivity
        self.rng = np.random.RandomState(random_seed)

        self.W_in = None
        self.W = None
        self.W_out = None
        self._fitted = False

    def _init_reservoir(self):
        """Initialize input and reservoir weights (Erdos-Renyi style)."""
        # Input weights
        self.W_in = self.rng.uniform(-1, 1, (self.N, self.input_dim)) * self.sigma_in

        # Sparse reservoir
        W = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if self.rng.rand() < self.connectivity:
                    W[i, j] = self.rng.uniform(-1, 1)
        W = W / np.max(np.abs(np.linalg.eigvals(W))) * self.rho   # scale spectral radius
        self.W = W

    def _step(self, x_prev: np.ndarray, u: np.ndarray) -> np.ndarray:
        """One reservoir step with leaky integrator."""
        u = np.asarray(u).reshape(-1)
        x_tilde = np.tanh(self.W_in @ u + self.W @ x_prev)
        x = (1 - self.alpha) * x_prev + self.alpha * x_tilde
        return x

    def _collect_states(self, U: np.ndarray, washout: int = 50) -> np.ndarray:
        """Run open loop and return reservoir states after washout."""
        if self.W is None:
            self._init_reservoir()

        N_t = len(U)
        X = np.zeros((N_t, self.N))
        x = np.zeros(self.N)

        for t in range(N_t):
            x = self._step(x, U[t])
            X[t] = x

        return X[washout:]

    def train(
        self,
        forcing_train: np.ndarray,
        target_train: Optional[np.ndarray] = None,
        washout: int = 50,
    ):
        """
        Train the readout.

        If target_train is None, we do one-step-ahead self-prediction
        (standard for next-value forecasting of the forcing signal).
        """
        forcing_train = np.asarray(forcing_train).reshape(-1, self.input_dim)

        if target_train is None:
            target_train = forcing_train[1:]          # predict next step
            forcing_train = forcing_train[:-1]

        X = self._collect_states(forcing_train, washout=washout)
        Y = target_train[washout:].reshape(-1, self.input_dim)

        # Augment with bias
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])

        reg = Ridge(alpha=self.tikh, fit_intercept=False)
        reg.fit(X_aug, Y)
        self.W_out = reg.coef_.T
        self._fitted = True

    def predict(
        self,
        forcing_history: np.ndarray,
        horizon: int = 50,
        washout: int = 20,
    ) -> Tuple[np.ndarray, float]:
        """
        Predict future forcing and compute a simple risk score.

        Returns:
            predicted_forcing : (horizon,)
            risk_score        : scalar in [0,1] (probability of large spike)
        """
        if not self._fitted:
            raise RuntimeError("Call train() before predict()")

        history = np.asarray(forcing_history).reshape(-1, self.input_dim)
        x = np.zeros(self.N)

        # Warm up on history
        for u in history[-washout:]:
            x = self._step(x, u)

        preds = []
        current_u = history[-1]

        for _ in range(horizon):
            x = self._step(x, current_u)
            y = (np.hstack([x, 1.0]) @ self.W_out).item()
            preds.append(y)
            current_u = np.array([y])   # closed loop

        preds = np.array(preds)

        # Simple risk: fraction of predicted |forcing| above 2*std of history
        hist_std = np.std(history)
        threshold = 2.0 * hist_std if hist_std > 0 else 0.1
        risk = float(np.mean(np.abs(preds) > threshold))

        return preds, risk


# Convenience function for quick HAVOK forcing risk analysis
def quick_forcing_risk(
    forcing: np.ndarray,
    train_frac: float = 0.7,
    horizon: int = 30,
    reservoir_size: int = 150,
) -> dict:
    """
    One-liner helper: train on past forcing, predict future risk.
    """
    n_train = int(len(forcing) * train_frac)
    train = forcing[:n_train]
    test_start = forcing[n_train - 20 : n_train]  # small history for warm start

    model = ForcingRiskPredictor(reservoir_size=reservoir_size)
    model.train(train)

    pred, risk = model.predict(test_start, horizon=horizon)

    return {
        "predicted_forcing": pred,
        "regime_shift_risk": risk,
        "horizon": horizon,
    }
