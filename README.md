# HAVOK Regime-Shift Detector v0.7.0

**Turn chaos into early-warning signals.**

Implementation of the HAVOK algorithm from *"Chaos as an Intermittently Forced Linear System"* (Brunton et al., 2017).

Given any univariate time series, HAVOK extracts the hidden intermittent forcing signal that precedes sudden regime shifts (seizures, crashes, tipping points, etc.).

## What's New in v0.7.0 🚀

- **sklearn-compatible** `HavokEstimator` — `fit() / transform() / predict_risk() / score()`
- **User analysis tools** — `analyze()` one-liner, `batch_analyze()`, `suggest_and_explain()`
- **Confidence intervals** — bootstrap forcing CI + risk probability (not just binary)
- **4 differentiation methods** — finite diff, spline, total variation, gradient
- **GPU acceleration** — transparent CuPy fallback for SVD/lstsq
- **Cross-validation** — `cross_val_score_havok()` with true hold-out
- **Benchmark suite** — 5 datasets × 5 methods, ranking table
- **Export** — CSV/JSON/`.havok` serialization
- **146 tests** (including property-based Hypothesis)

## Quick Start

### 1. Install
```bash
pip install -e ".[dev]"
```

### 2. Lorenz demo
```bash
havok demo
```

### 3. Predict regime-shift risk
```bash
havok predict data.csv -c price --horizon 30
```

### 4. Edge-of-chaos analysis
```bash
havok chaos data.csv -c eeg
```

### 5. Interactive dashboard
```bash
streamlit run dashboard/app.py
```

## Core Pipeline

1. **Time-delay embedding** → Hankel matrix
2. **SVD** → eigen-time-delay coordinates
3. **Linear regression** → isolate intermittent forcing
4. **Thresholding / change-point** → regime-shift risk
5. **ESN prediction** → forecast future forcing (NEW)
6. **Edge-of-chaos scoring** → Lyapunov + critical slowing down (NEW)

## CLI Commands

| Command | Description |
|---------|-------------|
| `havok demo` | Run HAVOK on Lorenz attractor |
| `havok analyze data.csv -c col` | Full analysis + interactive HTML report |
| `havok predict data.csv -c col` | ESN prediction of future forcing + risk |
| `havok chaos data.csv -c col` | Edge-of-chaos metrics |
| `havok suggest data.csv -c col` | Auto-tune tau/m parameters |

## ML Risk Predictor (NEW)
After HAVOK extracts the forcing signal, you can train a lightweight Echo State Network (ESN) to forecast future forcing values and estimate regime-shift risk.

```python
from havolib.ml_risk_predictor import quick_forcing_risk

result = quick_forcing_risk(forcing_signal, horizon=30)
print("Regime shift risk:", result["regime_shift_risk"])
```

The implementation is inspired by the excellent MagriLab ESN tutorials (leaky integrator, spectral radius scaling, Ridge readout).

See `havolib/ml_risk_predictor.py` and `references/extracted/esn_blueprint_from_magrilab.md`.

## Related Work & Insights
None of the following projects implement HAVOK directly, but they offer extremely valuable patterns:

- **Edge of Chaos theory** (vandijklab) → narrative for why forcing spikes matter
- **ESN + LSTM blueprints** (MagriLab/Tutorials) → basis for the ML risk predictor above
- **Benchmarking framework** (wangcaidao) → future validation harness
- **EEG / wearable motivation** (jobInregina/Chaos) → primary medical use-case

Full extraction and mapping: `references/github_insights.md`

## Project Structure
```
havok-toolbox/
├── pyproject.toml
├── engine.yaml            # Streaming engine config
├── havolib/
│   ├── config.py           # YAML profile loader
│   ├── pipeline.py         # Batch HAVOK
│   ├── embedding.py / decomposition.py / forcing.py / detection.py
│   ├── pre_processing.py / surrogate.py / auto_tune.py
│   ├── visualization.py
│   ├── ml_risk_predictor.py   # ESN for forcing prediction
│   ├── edge_of_chaos.py       # LLE + CSD + edge score
│   └── engine/                # Streaming engine
│       ├── ring_buffer.py / incremental_hankel.py / brand_svd.py
│       ├── incremental_havok.py / risk_engine.py / alert_pipeline.py
│       └── engine.py          # Async orchestrator
├── _cli_havok.py          # `havok` CLI (7 commands)
├── benchmark/             # 5 datasets × 5 methods + rankings
├── dashboard/
│   ├── app.py             # Batch analysis dashboard
│   └── engine_dashboard.py # Live streaming dashboard
├── data/                  # Sample CSVs + EDF
├── tests/                 # 146 tests (property-based + Hypothesis) — v0.7.0 adds Adaptive HAVOK, Hybrid Transformer, Federated Learning, Attribution, Arena
└── .github/workflows/     # CI
```

## Key Parameters
- `tau`: time delay for embedding
- `m`: embedding dimension (columns in Hankel matrix)
- `r`: number of retained modes (usually 3–10)
- `threshold_std` / `window`: control sensitivity of the risk detector

## References
- Brunton, Brunton, Proctor, Kutz. “Chaos as an Intermittently Forced Linear System.” *Nature Communications* 2017.
- Takens’ embedding theorem

## License
MIT. Build the revolution.
