"""
HAVOK Regime-Shift Detector
Core library for extracting intermittent forcing signals from time series
using time-delay embedding + SVD (Brunton et al. 2017).
"""

from .config import *
from .data_loader import (
    load_csv,
    generate_lorenz,
    download_chb_sample,
    load_eeg,
    load_chb_channel,
    generate_eeg_like,
    list_edf_channels,
)
from .embedding import hankel_matrix, auto_tau
from .decomposition import eigen_time_delay
from .forcing import extract_forcing
from .detection import threshold_risk, bayesian_changepoint
from .pipeline import HavokPipeline
from .visualization import plot_dashboard

__version__ = "0.2.0"
