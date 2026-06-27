# ESN Blueprint extracted from MagriLab/Tutorials

Source: https://github.com/MagriLab/Tutorials/tree/master/esn

## Core Architecture

The Echo State Network (ESN) implemented in `havolib/ml_risk_predictor.py` follows
the MagriLab patterns:

### Reservoir Initialization
- Sparse Erdos-Renyi random weights
- Spectral radius scaling: `W = W / max(|eigvals(W)|) * rho`
- Input weights: uniform [-1, 1] scaled by `input_scaling`

### State Update (Leaky Integrator)
```python
x_tilde = tanh(W_in @ u + W @ x_prev)
x = (1 - alpha) * x_prev + alpha * x_tilde
```

### Training
- Washout period (discard first N states)
- Open-loop state collection
- Ridge regression (Tikhonov) readout

### Key Hyperparameters
| Param | MagriLab | Our implementation |
|-------|----------|-------------------|
| reservoir_size | 500-1000 | 80-200 (lightweight) |
| spectral_radius | 0.9-1.2 | 0.95 default |
| leak_factor (alpha) | 0.1-0.5 | 0.3 default |
| input_scaling | 0.5-2.0 | 1.0 default |
| tikhonov (ridge alpha) | 1e-8 | 1e-8 |
| connectivity | 0.1 | 0.1 |

### Extensions in Our Implementation
- Vectorized state collection (vs original Python loop)
- Closed-loop prediction for forecasting
- Risk score: fraction of predicted |forcing| > 2*std(history)
- `quick_forcing_risk()` convenience wrapper
