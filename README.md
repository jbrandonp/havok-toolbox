# HAVOK Regime-Shift Detector

**Turn chaos into early-warning signals.**

Implementation of the HAVOK algorithm from *“Chaos as an Intermittently Forced Linear System”* (Brunton et al., 2017).

Given any univariate time series, HAVOK extracts the hidden intermittent forcing signal that precedes sudden regime shifts (seizures, crashes, tipping points, etc.).

## The "Edge of Chaos" Interpretation
Research on complex systems (see vandijklab / arXiv:2410.02536) shows that rich, adaptive behavior emerges at the **edge of chaos** — the narrow boundary between rigid order and pure randomness.

A strong HAVOK forcing spike can be read as the system **approaching or crossing this edge**. The forcing signal therefore serves as a direct, quantitative measure of how close the dynamics are to a regime shift.

This gives the toolbox both rigorous mathematics and a compelling scientific narrative.

## Quick Start (Portable)

### 1. Install (recommended)
```bash
# from the project folder
pip install -e .[dev]
```

This makes the `havok` command available globally (works from any directory).

### 2. Run the Lorenz demo (best first experience)
```bash
havok demo
```

Open `lorenz_demo.html` in your browser. You will see the forcing signal spiking right before chaotic bursts.

### 3. Analyze your own CSV
```bash
havok analyze your_data.csv --column "price" --output report.html
```

### 4. Launch the interactive dashboard
```bash
streamlit run dashboard/app.py
```

(From the project folder, or pass the full path to app.py after install.)

## Core Pipeline

1. **Time-delay embedding** → Hankel matrix
2. **SVD** → eigen-time-delay coordinates
3. **Linear regression** → isolate intermittent forcing
4. **Thresholding / change-point** → regime-shift risk

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
├── pyproject.toml         # Packaging (pip install -e .)
├── havolib/               # Core library (import havolib)
│   ├── embedding.py
│   ├── decomposition.py
│   ├── forcing.py
│   ├── detection.py
│   ├── pipeline.py
│   ├── visualization.py
│   └── ml_risk_predictor.py   # ESN-style predictor for forcing (NEW)
├── cli.py                 # Exposed as `havok` command after install
├── dashboard/
│   └── app.py
├── data/                  # Sample CSVs + EDF (used by defaults)
├── references/
│   └── github_insights.md
└── tests/
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
