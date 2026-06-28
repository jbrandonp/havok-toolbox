"""
HAVOK Regime-Shift Detector — time series analysis with early-warning regime-shift detection.
Core library for extracting intermittent forcing signals from time series
using time-delay embedding + SVD (Brunton et al. 2017).

Version 0.8.0 — Probabilistic risk (GEV), bootstrap CI, auto-rank, online BOCPD + Koopman adaptive, uncertainty module.
"""

from .config import (
    DEFAULT_TAU, DEFAULT_M, DEFAULT_R, DEFAULT_THRESHOLD_STD, DEFAULT_WINDOW,
    load_config, list_profiles, load_profile,
    HavokParams, PreprocessingConfig, PipelineConfig, EngineConfig, StreamConfig,
)
from .auto_tune import optimal_tau_mi, optimal_m_fnn, optimal_m_havok, suggest_parameters
from .data_loader import (
    load_csv, generate_lorenz, download_chb_sample, load_eeg,
    load_chb_channel, generate_eeg_like, list_edf_channels,
)
from .embedding import hankel_matrix, auto_tau
from .decomposition import eigen_time_delay
from .forcing import extract_forcing
from .detection import threshold_risk, pelt_changepoint, bayesian_changepoint  # bayesian_changepoint is deprecated alias
from .pipeline import HavokPipeline
from .visualization import plot_dashboard
from .ml_risk_predictor import ForcingRiskPredictor, quick_forcing_risk
from .edge_of_chaos import (
    largest_lyapunov_exponent, correlation_dimension,
    critical_slowing_down, edge_of_chaos_score,
)
from .engine import (
    RingBuffer, IncrementalHankel, BrandSVD, IncrementalHAVOK,
    RiskEngine, RiskLevel, AlertPipeline, AlertRule, AlertTarget, AlertLevel, HavokEngine,
    EngineStream, EngineRuntime,  # runtime config containers (not the frozen config.py ones)
)
from .gpu import is_gpu_available, svd, lstsq, norm
from .serialize import save_pipeline, load_pipeline
from .logging_config import init_logging, get_logger
from .estimator import HavokEstimator, cross_val_score_havok, DIFF_METHODS
from .user import analyze, AnalysisReport, batch_analyze, suggest_and_explain
from .multichannel import MultichannelHAVOK, MultichannelResult
from .automl import auto_optimize, suggest_from_automl
from .polars_loader import load_csv_fast, load_parquet_fast, batch_load_csvs
from .adaptive import AdaptiveHAVOK, AdaptiveResult, BayesianOnlineCP, RegimeMemory
from .attribution import explain_forcing_spike
from .hybrid import HavokTransformer
from .federated import FederatedHAVOK, FederatedModel
from .arena import BenchmarkArena, run_arena
from .uncertainty import (
    phase_randomized_surrogate, generate_surrogates, block_bootstrap, crps, conformal_interval
)

__version__ = "0.3.0"  # Portable install + pipeline/estimator unified + SVD auto-tune
