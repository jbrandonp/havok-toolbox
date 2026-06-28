# HAVOK Regime-Shift Detector v0.9.0

> **2026 Q2 hardening**: Portable install (pip-ready), production-grade robustness fixes across all modules, SVD-spectrum auto-tune replacing broken FNN, config unification, 0 runtime crashes on edge cases.

<p align="center">
  <img src="https://img.shields.io/badge/version-0.9.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.9+-green" alt="Python">
  <img src="https://img.shields.io/badge/tests-284%20passed-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/coverage-73%25-yellow" alt="Coverage">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/pip%20install-ready-success" alt="pip install">
</p>

**Turn chaos into actionable early-warning signals.**

`havok-toolbox` is the most complete open-source implementation of the **HAVOK** (Hankel Alternative View of Koopman) algorithm from *"Chaos as an Intermittently Forced Linear System"* (Brunton, Brunton, Proctor & Kutz, *Nature Communications*, 2017).

Given any univariate time series, HAVOK extracts the hidden **intermittent forcing signal** that precedes sudden regime shifts — seizures, market crashes, climate tipping points, industrial failures — before they happen.

---

## ✨ Features

| Category | Capability |
|----------|------------|
| **Core HAVOK** | Full Hankel embedding → SVD → eigen-time-delay → forcing extraction → regime-shift risk |
| **Adaptive** | Auto-detects regime transitions, adapts parameters per segment *(unique — no other HAVOK lib does this)* |
| **Multichannel** | Parallel multi-signal analysis (EEG 23ch, multi-asset, multi-sensor) with joint forcing + coupling |
| **AutoML** | Optuna TPE hyperparameter optimization — finds optimal τ, m, r automatically |
| **Hybrid ML** | HAVOK-Transformer (Neural ODE), ESN predictor, edge-of-chaos scoring (LLE + CSD) |
| **Federated** | Privacy-preserving training across hospitals/institutions (FedAvg + Differential Privacy) |
| **Explainable** | Forcing Attribution — explains *why* a spike occurred (amplitude, frequency, trend, noise) |
| **Benchmark** | 5 datasets × 5 methods, public leaderboard JSON generator |
| **Production** | sklearn-compatible `HavokEstimator`, GPU acceleration (CuPy), 10-50× Polars loader |
| **Streaming** | MQTT + CSV + Synthetic engine with alert pipeline (cooldown, dedup, webhook-ready) |
| **Export** | CSV / JSON / `.havok` serialization, Plotly visualization, Streamlit dashboard |

---

## 🚀 Quick Start

```bash
# Install
pip install havok-toolbox[all]

# Run the Lorenz demo in 1 line
havok demo

# Analyze a CSV file
havok analyze data.csv -c price

# Detect edge-of-chaos
havok chaos data.csv -c eeg

# Auto-optimize parameters with AutoML
havok suggest data.csv -c signal

# Run benchmark arena
havok benchmark
```

Or from Python:

```python
import numpy as np
from havolib import HavokEstimator, analyze

# 1-liner sklearn-compatible
forcing, risk = HavokEstimator(m=50, r=5).fit_transform(my_data)

# Full analysis with confidence intervals
report = analyze(my_eeg_data, bootstrap_ci=True)
print(report.summary())
report.export("results.csv")

# Multichannel (EEG, multi-asset)
from havolib import MultichannelHAVOK
mh = MultichannelHAVOK(n_channels=8)
result = mh.fit_transform(eeg_8ch)

# AutoML
from havolib import auto_optimize
best = auto_optimize(my_data, n_trials=100)
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
pip install havok-toolbox[fast]        # Polars (10x CSV)
pip install havok-toolbox[eeg]         # EDF/MNE support
pip install havok-toolbox[torch]       # HAVOK-Transformer
pip install havok-toolbox[all]         # Everything
pip install havok-toolbox[dev]         # Tests + Hypothesis

# From source
git clone https://github.com/jbrandonp/havok-toolbox
cd havok-toolbox && pip install -e ".[all]"
```

---

## 🧠 Algorithm

HAVOK decomposes a chaotic signal into **deterministic linear dynamics** + **intermittent forcing**:

1. **Time-delay embedding** → Hankel matrix H
2. **Truncated SVD** → eigen-time-delay coordinates V(t)
3. **Linear regression** V̇ ≈ AV → isolate forcing F(t) = V̇ − AV
4. **Thresholding** → regime-shift risk (binary or probabilistic)

The forcing signal spikes **before** the raw signal shows any visible change, making HAVOK a powerful **early warning system** for sudden regime transitions.

---

## 📁 Project Structure

```
havok-toolbox/
├── havolib/                    # Core library (30+ modules)
│   ├── pipeline.py             # Batch HAVOK pipeline (primary user entry)
│   ├── estimator.py            # sklearn-compatible HavokEstimator
│   ├── adaptive.py             # Non-stationary adaptive HAVOK + BOCPD
│   ├── multichannel.py         # Parallel multi-signal mHAVOK
│   ├── hybrid.py               # HAVOK-Transformer (Neural ODE)
│   ├── federated.py            # Federated Learning + DP
│   ├── attribution.py          # Forcing spike explanation
│   ├── automl.py               # Optuna hyperparameter optimization
│   ├── arena.py                # Public benchmark leaderboard
│   ├── edge_of_chaos.py        # LLE + CSD + edge score
│   ├── ml_risk_predictor.py    # ESN forcing forecaster
│   ├── uncertainty.py          # Surrogates, CRPS, conformal intervals
│   ├── engine/                 # Streaming engine (7 modules)
│   │   ├── engine.py           # Async orchestrator (MQTT/CSV/Synthetic)
│   │   ├── ring_buffer.py      # O(1) circular buffer
│   │   ├── incremental_havok.py# Sliding-window HAVOK
│   │   └── risk_engine.py      # Multi-dim risk scoring
│   ├── benchmark/              # 5 datasets × 5 methods (now inside havolib)
│   ├── dashboard/              # Streamlit dashboards v3 unified
│   │   ├── app.py              # Batch analysis
│   │   ├── engine_dashboard.py # Streaming engine
│   │   ├── advanced.py         # Comparison + What-if
│   │   └── v3.py               # Unified (multi + adaptive + attribution)
│   ├── gpu.py                  # GPU acceleration (CuPy)
│   ├── config.py               # Dataclass config + YAML profiles
│   ├── serialize.py            # .havok file format
│   ├── polars_loader.py        # 10-50× Pandas CSV/Parquet
│   ├── user.py                 # analyze/batch/bootstrap/export
│   ├── logging_config.py       # Structured logging
│   └── visualization.py        # Plotly figures
├── tests/                      # 284 tests (24 files)
│   ├── test_master_full.py     # Master suite (61 tests)
│   ├── test_v070_modules.py    # v0.7.0 module coverage
│   ├── test_properties.py      # Hypothesis property-based
│   ├── test_regression.py      # Golden value stability
│   ├── test_cli.py             # CLI integration
│   └── ...                     # 18 more test files
├── docs/
│   ├── adr.md                  # Architecture Decision Records
│   └── competitive_comparison.md
├── _cli_havok.py               # CLI (7 commands)
├── havok_config.yaml           # HAVOK profiles (EEG, finance, climate, Lorenz)
├── engine.yaml                 # Streaming engine config
├── pyproject.toml
└── README.md
```

---

## 🏆 Competitive Comparison

| Criterion | **havok-toolbox** | pykoopman | PyDMD | rhavok | deeptime |
|-----------|:---:|:---:|:---:|:---:|:---:|
| HAVOK implementation | ★★★★★ | ★★ | — | ★★★★ | ★★ |
| Adaptive/Non-stationary | ✅ | ❌ | ❌ | ❌ | ❌ |
| Multichannel (mHAVOK) | ✅ | ❌ | ❌ | ❌ | ❌ |
| AutoML (Optuna) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Explainability | ✅ | ❌ | ❌ | ❌ | ❌ |
| Federated Learning | ✅ | ❌ | ❌ | ❌ | ❌ |
| sklearn-compatible | ✅ | ✅ | ❌ | ❌ | ✅ |
| GPU acceleration | ✅ | ❌ | ❌ | ❌ | ❌ |
| Streaming engine | ✅ | ❌ | ❌ | ❌ | ❌ |
| Benchmark suite | ✅ | ❌ | ❌ | ❌ | ❌ |
| Dashboard | ✅ | ❌ | ❌ | ❌ | ❌ |
| Tests | 284 | ~20 | ~10 | ~5 | ~50 |

---

## ⚙️ CLI Reference

| Command | Description |
|---------|-------------|
| `havok demo` | Run HAVOK on Lorenz attractor |
| `havok analyze <file> -c <col>` | Full analysis + report |
| `havok suggest <file> -c <col>` | Auto-tune τ, m parameters |
| `havok predict <file> -c <col> --horizon 30` | ESN forcing prediction |
| `havok chaos <file> -c <col>` | Edge-of-chaos metrics |
| `havok benchmark` | Run benchmark arena |
| `havok engine` | Streaming engine control |

---

## 🔬 Key Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `τ` (tau) | Time delay for embedding | 1–30 |
| `m` | Embedding dimension (Hankel columns) | 10–100 |
| `r` | Truncated SVD rank | 2–15 |
| `threshold_std` | Risk detection sensitivity | 1.5–5.0 |
| `window` | Rolling window for risk | 20–300 |
| `diff_method` | Differentiation: `finite_diff`, `spline`, `total_variation`, `gradient` | — |

---

## 📊 Dashboard

Launch the interactive dashboard:

```bash
# Unified dashboard (multichannel + adaptive + attribution)
streamlit run dashboard/v3.py

# Advanced (comparison + what-if simulation)
streamlit run dashboard/advanced.py

# Streaming engine monitor
streamlit run dashboard/engine_dashboard.py
```

---

## 🤝 Contributing

Contributions welcome! See [ADRs](docs/adr.md) for architectural decisions and design philosophy.

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

---

## 🚀 What's New (v0.9 — Production Hardening)

- **Auto-tune fixed**: `optimal_m_havok()` uses SVD spectrum instead of broken FNN. `suggest_parameters()` now returns m ≥ 15 and caps tau ≤ 10 for meaningful forcing residuals.
- **Portable install**: `pip install havok-toolbox` works. `benchmark/` and `dashboard/` moved into `havolib/` for zero-config imports. `importlib.resources` for data files.
- **Robustness sweep**: `correlation_dimension` validates input ranges, `AdaptiveHAVOK` handles short data gracefully, `plot_dashboard` works with r<3, `FederatedHAVOK` raises `ValueError` instead of silent failure.
- **Naming cleanup**: `bayesian_changepoint` → `pelt_changepoint` (with deprecated alias). `_collect_states_vectorized` → `_collect_states` (wasn't vectorized). Engine uses `EngineStream`/`EngineRuntime` distinct from frozen config dataclasses.
- **Dead code removed**: vestigial `risk_head` in `hybrid.py`, buggy GEV tail-model branch in `_compute_gev_risk`.
- **Config unified**: no more name collision between engine's runtime config and `config.py`'s frozen dataclasses.
- **Tests**: 284 passed (was 276), 0 skipped, 73% coverage, `torch` + `pytest-cov` installed.

## 🚀 What's New (v0.8 — Core Deepening)

- **estimator.py**: GEV calibrated `risk_proba_`, `fit_with_ci()` (phase-randomized bootstrap), Gavish-Donoho optimal rank (`r='auto'`), **fixed real `transform(X)`** for new data + full sklearn compliance.
- **adaptive.py**: `BayesianOnlineCP` (Adams-MacKay streaming), `_koopman_drift_detect`, `RegimeMemory` (parameter meta-learning), soft regime blending.
- New `uncertainty.py`: surrogates, block bootstrap, CRPS, conformal helpers.
- Validated demos committed under `demo/` (Lorenz + reports).
- Personal notes cleaned; project now focused on verifiable numerical depth.

## 📄 License

MIT + Commons Clause — build the revolution.

---

<p align="center">
  <sub>Built by <a href="https://github.com/jbrandonp">Brandon</a> with ❤️ and chaos theory.</sub>
</p>
