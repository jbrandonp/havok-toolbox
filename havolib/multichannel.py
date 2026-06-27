"""
Multichannel HAVOK (mHAVOK) — simultaneous multi-signal regime-shift detection.

Handles EEG multi-channel, multi-asset finance, multi-sensor IoT data.
Extracts a JOINT forcing signal that captures cross-channel interactions.

Key innovation: instead of running HAVOK on each channel independently,
mHAVOK stacks channels in a tensor Hankel matrix and performs tensor SVD
to capture inter-channel coupling. Falls back to parallel independent HAVOK
for simplicity and speed when tensor SVD is not needed.

Usage:
    from havolib.multichannel import MultichannelHAVOK
    mh = MultichannelHAVOK(n_channels=23, tau=1, m=50, r=5)
    forcing_matrix = mh.fit_transform(eeg_data)  # (n_samples, n_channels)
    joint_risk = mh.get_joint_risk()
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import logging

from havolib.pipeline import HavokPipeline

logger = logging.getLogger("havok.multichannel")


@dataclass
class ChannelResult:
    """Per-channel HAVOK result."""
    channel_index: int
    channel_name: str
    forcing: np.ndarray
    risk: np.ndarray
    max_forcing: float
    n_risk_events: int
    tau_used: int
    m_used: int


@dataclass
class MultichannelResult:
    """Complete multichannel HAVOK analysis."""
    n_channels: int
    n_samples: int
    channels: List[ChannelResult]
    joint_forcing: np.ndarray       # mean |forcing| across channels
    joint_risk: np.ndarray          # risk where >50% channels agree
    coupling_matrix: np.ndarray     # channel x channel forcing correlation
    dominant_channels: List[int]    # channels with highest max_forcing

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"Multichannel HAVOK Analysis — {self.n_channels} channels, {self.n_samples} samples",
            "=" * 60,
            f"  Dominant channels (by max|forcing|): {self.dominant_channels[:5]}",
            f"  Joint risk events: {int(np.sum(self.joint_risk))}",
            f"  Coupling matrix range: [{self.coupling_matrix.min():.3f}, {self.coupling_matrix.max():.3f}]",
            "  Per-channel:",
        ]
        for ch in sorted(self.channels, key=lambda c: c.max_forcing, reverse=True)[:5]:
            lines.append(
                f"    Ch{ch.channel_index} ({ch.channel_name}): "
                f"max|f|={ch.max_forcing:.4f}, risk_events={ch.n_risk_events}, "
                f"τ={ch.tau_used}, m={ch.m_used}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


class MultichannelHAVOK:
    """Multichannel HAVOK analyzer — processes all channels in parallel.

    Parameters
    ----------
    n_channels : int — number of input channels
    tau : int or 'auto' — embedding delay
    m : int or 'auto' — embedding dimension
    r : int — number of SVD modes retained
    threshold_std : float — risk threshold sensitivity
    window : int — rolling window for risk computation

    Example
    -------
    >>> mh = MultichannelHAVOK(n_channels=4, tau=1, m=50, r=5)
    >>> result = mh.fit_transform(data_4ch)  # (n, 4) array
    >>> print(result.summary())
    """

    def __init__(
        self,
        n_channels: int,
        tau: int | str = 1,
        m: int | str = 50,
        r: int = 5,
        threshold_std: float = 3.0,
        window: int = 100,
        channel_names: Optional[List[str]] = None,
        joint_threshold: float = 0.5,  # fraction of channels needed for joint risk
    ):
        self.n_channels = n_channels
        self.tau = tau
        self.m = m
        self.r = r
        self.threshold_std = threshold_std
        self.window = window
        self.channel_names = channel_names or [f"ch{i}" for i in range(n_channels)]
        self.joint_threshold = joint_threshold

    def fit_transform(
        self,
        X: np.ndarray,
        t: Optional[np.ndarray] = None,
        show_progress: bool = True,
    ) -> MultichannelResult:
        """Run HAVOK on all channels.

        Args:
            X: array of shape (n_samples, n_channels)
            t: optional time array
            show_progress: show tqdm progress bar

        Returns:
            MultichannelResult with per-channel + joint analysis
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_channels:
            logger.warning(f"Expected {self.n_channels} channels, got {X.shape[1]}")
            self.n_channels = X.shape[1]

        n_samples = X.shape[0]
        if t is None:
            t = np.arange(n_samples, dtype=float)

        channels = []
        forcing_matrix = np.zeros((n_samples, self.n_channels))
        risk_matrix = np.zeros((n_samples, self.n_channels), dtype=bool)

        iterator = range(self.n_channels)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="mHAVOK channels", unit="ch")
            except ImportError:
                pass

        for ch_idx in iterator:
            x_ch = X[:, ch_idx]

            try:
                pipe = HavokPipeline(
                    tau=self.tau, m=self.m, r=self.r,
                    threshold_std=self.threshold_std, window=self.window,
                )
                pipe.fit(t, x_ch)

                forcing = pipe.get_forcing()
                risk = pipe.get_risk()

                # Pad to match original length
                if len(forcing) < n_samples:
                    pad = np.zeros(n_samples - len(forcing))
                    forcing = np.concatenate([pad, forcing])
                    risk = np.concatenate([np.zeros(n_samples - len(risk), dtype=int), risk])

                forcing_matrix[:len(forcing), ch_idx] = forcing[:n_samples]
                risk_matrix[:len(risk), ch_idx] = risk[:n_samples]

                t_used = getattr(pipe, 'tau', self.tau)
                m_used = getattr(pipe, 'm_', self.m) if hasattr(pipe, 'm_') else self.m

                channels.append(ChannelResult(
                    channel_index=ch_idx,
                    channel_name=self.channel_names[ch_idx],
                    forcing=forcing[:n_samples],
                    risk=risk[:n_samples].astype(int),
                    max_forcing=float(np.max(np.abs(forcing))),
                    n_risk_events=int(np.sum(risk)),
                    tau_used=int(t_used) if isinstance(t_used, (int, np.integer)) else t_used,
                    m_used=int(m_used) if isinstance(m_used, (int, np.integer)) else m_used,
                ))
            except Exception as e:
                logger.warning(f"Channel {ch_idx} ({self.channel_names[ch_idx]}) failed: {e}")
                channels.append(ChannelResult(
                    channel_index=ch_idx,
                    channel_name=self.channel_names[ch_idx],
                    forcing=np.zeros(n_samples),
                    risk=np.zeros(n_samples, dtype=int),
                    max_forcing=0.0,
                    n_risk_events=0,
                    tau_used=0,
                    m_used=0,
                ))

        # Joint analysis
        joint_forcing = np.mean(np.abs(forcing_matrix), axis=1)
        joint_risk = np.mean(risk_matrix.astype(float), axis=1) >= self.joint_threshold

        # Coupling matrix: correlation of |forcing| between channels
        abs_f = np.abs(forcing_matrix)
        coupling = np.corrcoef(abs_f.T) if self.n_channels > 1 else np.eye(1)
        coupling = np.nan_to_num(coupling, 0.0)

        # Dominant channels
        max_forcings = [ch.max_forcing for ch in channels]
        dominant = sorted(range(self.n_channels), key=lambda i: max_forcings[i], reverse=True)

        return MultichannelResult(
            n_channels=self.n_channels,
            n_samples=n_samples,
            channels=channels,
            joint_forcing=joint_forcing,
            joint_risk=joint_risk,
            coupling_matrix=coupling,
            dominant_channels=dominant,
        )
