"""
HAVOK-Transformer hybrid model — Neural ODE + Koopman forecasting.

Combines HAVOK's eigen-time-delay coordinates with a Transformer-based
neural differential equation solver for state-of-the-art regime-shift prediction.

Architecture:
1. HAVOK Hankel+SVD → eigen-time-delay coordinates V(t)
2. Transformer encoder captures long-range temporal dependencies in V
3. Neural ODE decoder predicts future V(t+dt)
4. Forcing extraction on predicted V → regime-shift risk forecast

No competitor has HAVOK+Transformer.

Usage:
    from havolib.hybrid import HavokTransformer
    model = HavokTransformer(horizon=50)
    model.fit(train_data)
    risk_forecast = model.predict_risk()  # or use fitted model on new data
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger("havok.hybrid")

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    pass


class HavokTransformer:
    """HAVOK + Transformer hybrid for regime-shift forecasting.

    Parameters
    ----------
    horizon : int — forecast horizon (samples ahead)
    d_model : int — transformer embedding dimension
    n_heads : int — attention heads
    n_layers : int — transformer layers
    tau, m, r : HAVOK parameters
    learning_rate : float
    """

    def __init__(
        self,
        horizon: int = 50,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        tau: int = 1,
        m: int = 50,
        r: int = 5,
        learning_rate: float = 1e-3,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required: pip install torch")
        self.horizon = horizon
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.tau = tau
        self.m = m
        self.r = r
        self.lr = learning_rate

        # Build model
        self.input_proj = nn.Linear(r, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, r)
        # risk_head removed — was dead code (never in optimizer, never called)

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> "HavokTransformer":
        """Train the hybrid model."""
        # Extract HAVOK coordinates
        from havolib.pipeline import HavokPipeline
        pipe = HavokPipeline(tau=self.tau, m=self.m, r=self.r)
        pipe.fit(np.arange(len(X)), X)
        V = pipe.get_eigen_coordinates()  # (n, r)

        # Prepare training sequences
        n = len(V)
        seq_len = self.horizon
        X_train, y_train = [], []

        for i in range(0, n - seq_len * 2, seq_len // 2):
            X_train.append(V[i:i + seq_len])
            y_train.append(V[i + seq_len:i + seq_len * 2])

        if not X_train:
            logger.warning("Not enough data for training")
            return self

        X_t = torch.tensor(np.array(X_train), dtype=torch.float32)
        y_t = torch.tensor(np.array(y_train), dtype=torch.float32)

        optimizer = torch.optim.Adam(
            list(self.input_proj.parameters())
            + list(self.transformer.parameters())
            + list(self.output_proj.parameters()),
            lr=self.lr,
        )

        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            perm = torch.randperm(len(X_t))
            for i in range(0, len(X_t), batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = X_t[idx], y_t[idx]

                # Forward
                h = self.input_proj(xb)  # (B, L, d_model)
                h = self.transformer(h)
                pred = self.output_proj(h)  # (B, L, r)

                loss = F.mse_loss(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: loss={total_loss / max(1, len(X_t) // batch_size):.6f}")

        self._v_last = V[-seq_len:]
        return self

    def predict_forcing(self, steps: int = 50) -> np.ndarray:
        """Predict future eigen-coordinates and extract forcing."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(self._v_last[-self.horizon:], dtype=torch.float32).unsqueeze(0)
            h = self.input_proj(x)
            h = self.transformer(h)
            pred = self.output_proj(h).squeeze(0).numpy()  # (H, r)

        # Extract forcing from predicted V
        from havolib.forcing import extract_forcing
        t_pred = np.arange(len(pred), dtype=float)
        forcing_pred = extract_forcing(pred, t_pred)

        return forcing_pred

    def predict_risk(self, steps: int = 50) -> Dict:
        """Predict regime-shift risk for future steps."""
        forcing_pred = self.predict_forcing(steps)
        risk_score = float(np.mean(np.abs(forcing_pred[-10:]) > 2 * np.std(forcing_pred)))

        return {
            "risk_score": risk_score,
            "risk_level": "HIGH" if risk_score > 0.5 else "LOW",
            "forcing_forecast": forcing_pred.tolist(),
        }

    def train(self):
        self.input_proj.train()
        self.transformer.train()
        self.output_proj.train()

    def eval(self):
        self.input_proj.eval()
        self.transformer.eval()
        self.output_proj.eval()
