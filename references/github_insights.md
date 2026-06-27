# GitHub Insights & Pattern Extraction for HAVOK Toolbox

This document captures actionable patterns extracted from key repositories relevant to HAVOK (intermittent forcing / regime-shift detection in chaotic and physiological time series).

## 1. vandijklab / edge-of-chaos (core narrative)
- Strong framing: "intelligence / computation lives at the edge of chaos" (arXiv:2410.02536).
- A strong HAVOK forcing spike can be read as the system **approaching or crossing this edge**.
- The forcing signal therefore serves as a direct, quantitative measure of how close the dynamics are to a regime shift.
- Recommended language for papers / READMEs: "HAVOK forcing as an edge-of-chaos detector".

## 2. MagriLab / Tutorials (ESN reservoir for ML risk predictor)
- Excellent clean ESN implementation (esn/esn.py).
- Key patterns extracted and implemented in `havolib/ml_risk_predictor.py`:
  - Leaky integrator: `x = (1-alpha)*x_prev + alpha * tanh(...)`
  - Sparse random reservoir with spectral radius scaling
  - Washout period + open-loop state collection
  - Ridge regression readout (Tikhonov)
- Hyperparameters: reservoir_size, spectral_radius, leak_factor, input_scaling, tikhonov, connectivity.
- Used to build `ForcingRiskPredictor` + `quick_forcing_risk` helper.
- Reference file: `references/extracted/esn_blueprint_from_magrilab.md`

## 3. wangcaidao / Chaos-Prediction (benchmarking framework)
- Clean train/eval pipelines for MLP_P2P and other predictors.
- Useful for future HAVOK vs. other methods comparison (MLP, LSTM, Seq2Seq, FNO).
- Metrics and train loops can be adapted for forcing prediction benchmarks.

## 4. jobInregina / Chaos (EEG / wearable motivation)
- Strong motivation: brain dynamics often sit "at the brink of chaos".
- Examples of modeling Rossler / Chua with ANNs for EEG-like signals.
- Good for real physiological data pipelines, artifact handling, multi-channel considerations.
- Goal alignment: portable / wearable regime-shift early warning.

## 5. mkrauth96 / forecasting (additional patterns)
- Reservoir computing + Lyapunov exponent ideas.
- Useful for future stability / predictability extensions.

## Action Items Derived
- [done] ML risk predictor using MagriLab ESN patterns.
- [in progress] Edge-of-chaos narrative in README.
- Future: full benchmark suite against wangcaidao-style predictors.
- Future: integrate jobInregina EEG channel selection / artifact rejection patterns into data_loader.
- Future: Lyapunov / recurrence quantification as complementary diagnostics.

All clones live in `external/` (shallow `--depth 1`).
