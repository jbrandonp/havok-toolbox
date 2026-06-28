# HAVOK Regime-Shift Detector v0.3.0

> Portable install, one-click UX for non-programmers, pipeline/estimator unified, production-ready robustness.

<p align="center">
  <img src="https://img.shields.io/badge/version-0.3.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.9+-green" alt="Python">
  <img src="https://img.shields.io/badge/tests-284%20passed-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/coverage-73%25-yellow" alt="Coverage">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/pip%20install-ready-success" alt="pip install">
</p>

**Turn chaos into actionable early-warning signals.⚡**

`🌀havok-toolbox` implements the **HAVOK** (Hankel Alternative View of Koopman) algorithm from *"Chaos as an Intermittently Forced Linear System"* (Brunton, Brunton, Proctor & Kutz, *Nature Communications*, 2017). Given a univariate time series, HAVOK extracts the hidden **intermittent forcing signal** that precedes sudden regime shifts — seizures in EEG🧠, market crashes📉, climate tipping points🌍, industrial failures⚙️ — before they manifest in the raw data.

---

## ✨ Features

| Category | Capability |
|----------|------------|
| **Core HAVOK** | Full pipeline: Hankel embedding → truncated SVD → eigen-time-delay coordinates → forcing extraction → regime-shift risk quantification |
| **Auto-tuning** | SVD-spectrum based `optimal_m_havok()` replaces FNN; Mutual Information delay selection with automatic tau capping |
| **sklearn API** | `HavokEstimator` with `fit()`, `transform()`, `fit_transform()`, `score()`, `get_params()` — compatible with `GridSearchCV` and `Pipeline` |
| **Adaptive** | Non-stationary analysis: BOCPD or PELT changepoint detection, per-segment parameter retuning, soft regime blending, RegimeMemory |
| **Multichannel** | Two modes: `parallel` (fast per-channel) and `composite` (true mHAVOK with joint Hankel SVD capturing cross-channel coupling) |
| **AutoML** | Optuna TPE hyperparameter optimization over (τ, m, r, threshold, window, diff_method) with median pruning |
| **Hybrid ML** | HAVOK-Transformer (PyTorch encoder-decoder on eigen-coordinates); ESN forcing forecaster |
| **Edge of Chaos** | Rosenstein Largest Lyapunov Exponent, Grassberger-Procaccia correlation dimension, critical slowing down, combined edge score |
| **Uncertainty** | Phase-randomized surrogate testing, block bootstrap confidence intervals, CRPS scoring, conformal prediction |
| **Federated** | Privacy-preserving multi-client aggregation with (ε, δ)-differential privacy for healthcare/institutional deployment |
| **Attribution** | Per-spike explanation: amplitude contribution, frequency shift, trend deviation, noise component |
| **Production** | GPU acceleration via CuPy; Polars CSV loader (10–50× faster than pandas); `.havok` serialization format |
| **Streaming** | Async engine with MQTT, CSV-watch, and synthetic sources; alert pipeline with cooldown and deduplication |
| **One-click app** | Streamlit dashboard with drag-and-drop file upload, auto-detection, one-click CSV/HTML report export — zero coding required |
| **Benchmark** | 5 datasets × 5 methods; Arena generates JSON leaderboard |

---

## 🚀 Quick Start

### No coding required
```bash
# Double-click on Windows
run_havok_app.bat

# Or from terminal
pip install havok-toolbox[app]
havok-app
```
Opens a browser with drag-and-drop file upload, auto-analysis, and report download.

### Command-line
```bash
pip install havok-toolbox

# Analyze a CSV file (auto-detects columns)
havok analyze data.csv

# Specify column
havok analyze data.csv -c price -o results.csv

# Run benchmark
havok benchmark

# Initialize streaming engine config
havok engine init
```

### Python API
```python
import numpy as np
from havolib import HavokPipeline, HavokEstimator

# One-liner with sklearn-compatible estimator
est = HavokEstimator(tau=1, m=50, r=5)
forcing = est.fit_transform(my_signal)  # returns forcing array

# Full pipeline with auto-tuning
pipe = HavokPipeline()
pipe.auto_fit(None, my_signal)
forcing = pipe.get_forcing()
risk = pipe.get_risk()

# Multichannel (EEG, multi-asset, sensor arrays)
from havolib import MultichannelHAVOK
mh = MultichannelHAVOK(n_channels=8, method="composite")
result = mh.fit_transform(eeg_data)  # (n_samples, n_channels)

# Adaptive non-stationary
from havolib import AdaptiveHAVOK
result = AdaptiveHAVOK().fit_transform(nonstationary_signal)

# High-level analysis with bootstrap confidence intervals
from havolib import analyze
report = analyze(eeg_signal, bootstrap_ci=True)
print(report.summary())
report.export("results.csv")
```

---

## 📦 Installation

```bash
# Base install
pip install havok-toolbox

# With optional extras
pip install havok-toolbox[streaming]   # MQTT engine
pip install havok-toolbox[gpu]         # CuPy acceleration
pip install havok-toolbox[automl]      # Optuna optimization
pip install havok-toolbox[fast]        # Polars (10-50× CSV loading)
pip install havok-toolbox[eeg]         # EDF/MNE support
pip install havok-toolbox[torch]       # HAVOK-Transformer
pip install havok-toolbox[app]         # Streamlit dashboard
pip install havok-toolbox[all]         # Everything
pip install havok-toolbox[dev]         # Tests + Hypothesis

# From source
git clone https://github.com/jbrandonp/havok-toolbox
cd havok-toolbox && pip install -e ".[dev]"
```

---

## 🧠 Algorithm

HAVOK decomposes a chaotic signal into **deterministic linear dynamics** + **intermittent forcing**:

1. **Time-delay embedding**: Build Hankel matrix **H** by sliding a window of size `m` with delay `τ` across the signal
2. **Truncated SVD**: Decompose H ≈ **U Σ Vᵀ** retaining `r` modes; eigen-time-delay coordinates **V(t)** capture the attractor geometry
3. **Linear model**: Fit V̇ ≈ **A V** via least squares; the residual **F(t) = V̇ − A V** is the intermittent forcing
4. **Risk detection**: Apply rolling thresholding on ‖F(t)‖ to flag regime shifts; probabilistic risk via percentile-calibrated logistic scaling

The forcing signal spikes **before** the raw signal shows any visible change, making HAVOK an effective early warning system for sudden regime transitions.

**Differentiation methods**: `finite_diff` (central differences, default), `spline_diff` (cubic spline via SciPy, noise-robust), `total_variation_diff` (TV-regularized, best for sharp jumps), `gradient` (NumPy wrapper).

---

## 📐 Paper Correspondence

This section maps every equation-level concept from Brunton et al. 2017 to
the code. Modules not listed here are **post-paper engineering extensions**.

### Core algorithm (exact compliance)

| Paper step | Equation / concept | Code | Status |
|---|---|---|---|
| 1. Delay embedding | H[k,i] = x[k + i·τ] | `embedding.py` → `hankel_matrix()` | ✅ Exact |
| 2. Truncated SVD | H ≈ U Σ V^T, keep r modes | `decomposition.py` → `eigen_time_delay()` | ✅ Exact¹ |
| 3. Linear model | v̇_r ≈ Σ a_i v_i + F(t) | `forcing.py` → `extract_forcing()` | ✅ Exact² |
| 4. Risk threshold | ‖F(t)‖ > threshold → risk=1 | `detection.py` → `threshold_risk()` | ✅ Aligned³ |

¹ Returns numpy's U[:, :r] (left singular vectors, time in rows). Paper calls these V(t). Naming convention difference only, not a mathematical deviation.
² Adds bias term fitting affine model. On zero-mean data: no difference. On uncentered data: absorbs constant offset.
³ Paper uses fixed global threshold; code uses rolling std — more robust for non-stationary forcing amplitude.

### Parameter tuning (practical defaults)

| Heuristic | Code | Justification |
|---|---|---|
| τ ≤ 10 cap | `auto_tune.py` | Keeps coordinates correlated for linear model. Override for slow signals. |
| m ≥ 15 floor | `auto_tune.py` | Below this, Koopman linear approximation unreliable. Empirical, not theoretical. |
| m ≈ m₉₉ × 3 | `auto_tune.py` → `optimal_m_havok()` | 99% SVD energy for attractor reconstruction (Takens); HAVOK needs more. Validated on Lorenz, EEG, finance. |

### Post-paper extensions

| Module | What it adds |
|---|---|
| `adaptive.py` | BOCPD / PELT changepoint, per-segment retuning |
| `multichannel.py` | Parallel (per-channel) + composite (joint Hankel SVD) modes |
| `automl.py` | Optuna TPE hyperparameter optimization |
| `hybrid.py` | PyTorch Transformer on eigen-coordinates |
| `federated.py` | FedAvg with (ε, δ)-differential privacy |
| `engine/` | Streaming MQTT/CSV engine with alert pipeline |
| `edge_of_chaos.py` | LLE, correlation dimension, critical slowing down |
| `surrogate.py`, `uncertainty.py` | Statistical validation and confidence intervals |
| `dashboard/`, `_cli_havok.py` | Streamlit dashboard, CLI |

### SVD solver equivalence

`solver="randomized"` (sklearn) vs exact SciPy SVD on Lorenz (3000 pts, m=50):
mean forcing difference < 10⁻⁶. Default `solver="auto"` prefers GPU (CuPy).

### Reproducibility

Golden-value regression tests verify Lorenz forcing stability across versions.
`generate_lorenz()` is deterministic — same seed = same trajectory.

---

## 📁 Project Structure

```
havok-toolbox/
├── havolib/                    # Core library (36 modules, 7,500+ lines)
│   ├── pipeline.py             # HavokPipeline — primary orchestration layer
│   ├── estimator.py            # HavokEstimator — sklearn BaseEstimator + TransformerMixin
│   ├── adaptive.py             # AdaptiveHAVOK — non-stationary with BOCPD + Koopman drift
│   ├── multichannel.py         # MultichannelHAVOK — parallel + composite modes
│   ├── hybrid.py               # HavokTransformer — PyTorch Transformer on eigen-coordinates
│   ├── federated.py            # FederatedHAVOK — FedAvg with differential privacy
│   ├── attribution.py          # explain_forcing_spike — per-feature spike explanation
│   ├── automl.py               # auto_optimize — Optuna TPE hyperparameter search
│   ├── arena.py                # BenchmarkArena — public leaderboard generator
│   ├── edge_of_chaos.py        # Rosenstein LLE, GP correlation dimension, CSD, edge score
│   ├── ml_risk_predictor.py    # FastForcingRiskPredictor — echo state network forecaster
│   ├── uncertainty.py          # Surrogates, block bootstrap, CRPS, conformal intervals
│   ├── surrogate.py            # Phase-randomized Fourier surrogates
│   ├── config.py               # Frozen dataclass config + YAML profiles (eeg, finance, climate, lorenz)
│   ├── data_loader.py          # generate_lorenz, load_csv, load_eeg with portable paths
│   ├── polars_loader.py        # load_csv_fast — Polars-accelerated CSV/Parquet loading
│   ├── pre_processing.py       # preprocess — Savitzky-Golay smoothing, IQR outlier removal, detrend
│   ├── serialize.py            # save_pipeline / load_pipeline — .havok binary format
│   ├── user.py                 # analyze, batch_analyze, bootstrap — high-level user API
│   ├── visualization.py        # plot_dashboard — Plotly 4-panel figure
│   ├── gpu.py                  # Transparent CuPy fallback for svd, lstsq, norm, eigvals
│   ├── logging_config.py       # init_logging — structured logging setup
│   ├── embedding.py            # hankel_matrix, auto_tau — delay embedding primitives
│   ├── decomposition.py        # eigen_time_delay — truncated SVD on Hankel
│   ├── forcing.py              # extract_forcing — linear model residual
│   ├── detection.py            # threshold_risk, pelt_changepoint — risk flagging
│   ├── auto_tune.py            # optimal_m_havok, optimal_tau_mi, suggest_parameters
│   ├── engine/                 # Streaming engine subsystem (7 modules)
│   │   ├── engine.py           # HavokEngine — async orchestrator (MQTT, CSV, synthetic)
│   │   ├── ring_buffer.py      # RingBuffer — O(1) circular buffer
│   │   ├── incremental_hankel.py # IncrementalHankel — streaming Hankel construction
│   │   ├── incremental_havok.py  # IncrementalHAVOK — sliding-window decomposition
│   │   ├── brand_svd.py        # BrandSVD — incremental SVD
│   │   ├── risk_engine.py      # RiskEngine — multi-dimensional risk scoring
│   │   └── alert_pipeline.py   # AlertPipeline — cooldown, dedup, webhook routing
│   ├── benchmark/              # 5 datasets × 5 methods (packaged with library)
│   │   ├── runner.py           # run_benchmark, print_summary
│   │   ├── baselines.py        # rolling_std, cusum, arima_residual detectors
│   │   └── cli.py              # Click CLI for benchmark
│   └── dashboard/              # Streamlit dashboards
│       ├── simple.py           # One-click app for non-programmers (NEW)
│       ├── app.py              # Batch analysis dashboard
│       ├── advanced.py         # Comparison + what-if simulation
│       ├── engine_dashboard.py # Streaming engine monitor
│       └── v3.py               # Unified (multichannel + adaptive + attribution)
├── tests/                      # 284 tests (24 files)
│   ├── test_master_full.py     # Master suite (61 tests)
│   ├── test_v070_modules.py    # Module integration tests
│   ├── test_properties.py      # Hypothesis property-based testing
│   ├── test_regression.py      # Golden value stability
│   ├── test_cli.py             # CLI integration
│   └── ...                     # 19 more test files
├── docs/
│   ├── adr.md                  # Architecture Decision Records
│   └── competitive_comparison.md
├── demo/                       # Validated Lorenz demo with reports
├── _cli_havok.py               # CLI entry point (havok analyze, benchmark, engine)
├── run_havok_app.bat           # Windows double-click launcher
├── havok_config.yaml           # Named profiles (eeg, finance, climate, lorenz_demo)
├── engine.yaml                 # Streaming engine configuration
├── pyproject.toml
└── README.md
```

---

## 🏆 Competitive Comparison

| Criterion | **havok-toolbox** | pykoopman | PyDMD | rhavok | deeptime |
|-----------|:---:|:---:|:---:|:---:|:---:|
| HAVOK fidelity | ★★★★★ | ★★ | — | ★★★★ | ★★ |
| Adaptive/Non-stationary | ✅ | ❌ | ❌ | ❌ | ❌ |
| Multichannel (true mHAVOK) | ✅ | ❌ | ❌ | ❌ | ❌ |
| AutoML (Optuna) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Explainability | ✅ | ❌ | ❌ | ❌ | ❌ |
| Federated learning | ✅ | ❌ | ❌ | ❌ | ❌ |
| sklearn-compatible | ✅ | ✅ | ❌ | ❌ | ✅ |
| GPU acceleration | ✅ | ❌ | ❌ | ❌ | ❌ |
| Streaming engine | ✅ | ❌ | ❌ | ❌ | ❌ |
| Benchmark suite | ✅ | ❌ | ❌ | ❌ | ❌ |
| Interactive dashboard | ✅ | ❌ | ❌ | ❌ | ❌ |
| One-click UX (no coding) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Tests | 284 | ~20 | ~10 | ~5 | ~50 |

---

## ⚙️ CLI Reference

| Command | Description |
|---------|-------------|
| `havok analyze <file> [-c COLUMN] [-o OUTPUT]` | One-click analysis with auto-tuning |
| `havok benchmark [--datasets X] [--methods Y]` | Run benchmark arena |
| `havok engine init` | Create default engine.yaml |
| `havok-app` | Launch Streamlit dashboard |

---

## 🔬 Key Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `τ` (tau) | Time delay for Hankel embedding | 1–30 |
| `m` | Embedding dimension (Hankel columns) | 10–100 |
| `r` | Truncated SVD rank | 2–15 |
| `threshold_std` | Risk detection sensitivity (standard deviations) | 1.5–5.0 |
| `window` | Rolling window for risk computation | 20–300 |
| `diff_method` | Differentiation: `finite_diff` (default), `spline`, `total_variation`, `gradient` | — |
| `method` | Multichannel mode: `parallel` (default) or `composite` (joint decomposition) | — |

---

## 📊 Dashboard

```bash
# One-click app (recommended for non-programmers)
streamlit run havolib/dashboard/simple.py

# Unified dashboard (multichannel + adaptive + attribution)
streamlit run havolib/dashboard/v3.py

# Advanced (comparison + what-if simulation)
streamlit run havolib/dashboard/advanced.py

# Streaming engine monitor
streamlit run havolib/dashboard/engine_dashboard.py
```

---

## 🔬 Validation & Testing

The test suite covers correctness, edge cases, and numerical stability:

- **284 tests**✅ passing with 73% line coverage
- **Property-based tests**🔁 via Hypothesis: SVD orthonormality, embedding isotonicity, forcing determinism
- **Golden value tests**🏅: fixed-seed Lorenz forcing output verified across versions
- **Edge case coverage**🛡️: empty signals, constant signals, NaN/Inf handling, very short data, single-channel, invalid parameters
- **Streaming engine**🌊: buffer overflow, incremental SVD stability, alert deduplication
- **Regression suite**🧪 (`test_master_full.py`): 61 tests across all subsystems

Run locally:
```bash
pip install havok-toolbox[dev]
pytest tests/ -v --cov=havolib
```

---

## 🤝 Contributing

Contributions welcome. See [Architecture Decision Records](docs/adr.md) for design philosophy and technical decisions.

```bash
git clone https://github.com/jbrandonp/havok-toolbox
cd havok-toolbox
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 📚 References

- Brunton, Brunton, Proctor, Kutz. *"Chaos as an Intermittently Forced Linear System."* Nature Communications, 2017. [DOI: 10.1038/s41467-017-00030-8](https://doi.org/10.1038/s41467-017-00030-8)
- Takens, F. *"Detecting strange attractors in turbulence."* Lecture Notes in Mathematics, 1981.
- Kutz, Brunton, Brunton, Proctor. *"Dynamic Mode Decomposition."* SIAM, 2016.
- Gavish & Donoho. *"The Optimal Hard Threshold for Singular Values."* IEEE Trans. Inf. Theory, 2014.
- Rosenstein, Collins, De Luca. *"A practical method for calculating largest Lyapunov exponents."* Physica D, 1993.

---

## 🚀 Changelog

### v0.3.0 — First Stable Release

- **Auto-tune fixed**: `optimal_m_havok()` uses SVD spectrum instead of broken FNN. `suggest_parameters()` returns m ≥ 15 with tau capped for meaningful forcing residuals.
- **Pipeline/estimator unified**: `HavokPipeline.fit()` delegates to `HavokEstimator` internally — single center of truth for HAVOK math.
- **True mHAVOK**: Added `method="composite"` to `MultichannelHAVOK` — composite Hankel matrix with joint SVD for genuine cross-channel coupling.
- **One-click UX**: `run_havok_app.bat` (Windows launcher) and `havok-app` CLI entry point. Drag-and-drop Streamlit dashboard with auto-detection and report export.
- **Portable install**: `pip install havok-toolbox` works from any directory. `benchmark/` and `dashboard/` moved into `havolib/`. `importlib.resources` for data files.
- **Robustness**: `correlation_dimension` validates input ranges; `AdaptiveHAVOK` handles short data gracefully; `plot_dashboard` works with r<3; `FederatedHAVOK` raises `ValueError` instead of silent failure.
- **Naming accuracy**: `bayesian_changepoint` → `pelt_changepoint` (with deprecated alias); `_collect_states_vectorized` → `_collect_states`; engine uses `EngineStream`/`EngineRuntime`.
- **Dead code removed**: vestigial `risk_head` in `hybrid.py`; buggy GEV tail-model branch in `_compute_gev_risk`.
- **Tests**: 284 passed (was 276), 0 skipped, 73% coverage. `torch` + `pytest-cov` installed.

---

## 📄 License

MIT + Commons Clause.

---

<p align="center">
  <sub>Built by <a href="https://github.com/jbrandonp">Brandon Palhano</a></sub>
</p>
