import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_dashboard(time: np.ndarray,
                   raw: np.ndarray,
                   forcing: np.ndarray,
                   risk: np.ndarray,
                   V: np.ndarray = None) -> go.Figure:
    """
    Interactive multi-panel dashboard for HAVOK analysis.
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.28, 0.22, 0.18, 0.32],
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "scene"}],
        ],
        subplot_titles=(
            "Original Time Series",
            "Intermittent Forcing Signal",
            "Regime-Shift Risk (binary)",
            "Attractor in Eigen-Time-Delay Coordinates (r≥3)"
        )
    )

    # 1. Raw signal
    fig.add_trace(
        go.Scatter(x=time, y=raw, mode='lines', name='Signal', line=dict(color='#1f77b4')),
        row=1, col=1
    )

    # 2. Forcing
    fig.add_trace(
        go.Scatter(x=time, y=forcing, mode='lines', name='Forcing', line=dict(color='#d62728')),
        row=2, col=1
    )

    # 3. Risk
    fig.add_trace(
        go.Scatter(x=time, y=risk, fill='tozeroy', name='Risk', line=dict(color='#ff7f0e')),
        row=3, col=1
    )

    # 4. Phase portrait (if possible)
    if V is not None and V.shape[1] >= 3:
        fig.add_trace(
            go.Scatter3d(
                x=V[:, 0], y=V[:, 1], z=V[:, 2],
                mode='lines',
                name='Attractor',
                line=dict(width=2, color='#2ca02c')
            ),
            row=4, col=1
        )
        fig.update_scenes(
            dict(xaxis_title='v1', yaxis_title='v2', zaxis_title='v3'),
            row=4, col=1
        )
    else:
        fig.add_annotation(
            text="Need r ≥ 3 for 3D attractor view",
            x=0.5, y=0.5, showarrow=False,
            row=4, col=1
        )

    fig.update_layout(
        height=1000,
        showlegend=False,
        title_text="HAVOK Regime-Shift Detector — Forcing & Risk Analysis",
        margin=dict(t=60, b=30)
    )
    return fig
