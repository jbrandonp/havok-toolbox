"""
HAVOK Dashboard v3 — All modules integrated.
streamlit run dashboard/v3.py
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time, tempfile, os, json

st.set_page_config(page_title="HAVOK Pro v3", layout="wide", page_icon="⚡")
st.title("⚡ HAVOK Pro v3 — Unified Dashboard")
st.caption("Multichannel · Adaptive · Attribution · Federated · Arena")

# ── Sidebar ──
with st.sidebar:
    st.header("📊 Data")
    mode = st.selectbox("Mode", ["Synthetic Lorenz", "Sine+Jump (2 regimes)", "Upload CSV"])
    n_points = st.slider("Points", 300, 5000, 1500)
    st.header("⚙️ Analysis")
    analysis_type = st.selectbox("Type", ["Quick (HAVOK basic)", "Adaptive (auto-detect regimes)",
                                          "Multichannel (if multi-col)", "Attribution (explain spikes)"])
    run = st.button("▶️ Run", type="primary", use_container_width=True)

if not run:
    st.info("Configure and click Run")
    st.stop()

# ── Generate data ──
t = np.arange(n_points, dtype=float)
if mode == "Synthetic Lorenz":
    from havolib.data_loader import generate_lorenz
    _, x = generate_lorenz(n_points=n_points)
elif mode == "Sine+Jump (2 regimes)":
    x = np.sin(np.linspace(0, 15*np.pi, n_points))
    x[n_points//2:] *= 3
    x += np.random.randn(n_points) * 0.2
else:
    st.info("Upload a CSV")
    st.stop()

# ── Run analysis ──
t0 = time.perf_counter()

if analysis_type == "Quick (HAVOK basic)":
    from havolib.estimator import HavokEstimator
    est = HavokEstimator(m=30, r=5).fit(x)
    forcing, risk = est.forcing_, est.risk_
    regimes = None
elif analysis_type == "Adaptive (auto-detect regimes)":
    from havolib.adaptive import AdaptiveHAVOK
    adp = AdaptiveHAVOK(min_segment_length=80)
    result = adp.fit_transform(x, show_progress=False)
    forcing, risk = result.full_forcing, result.full_risk
    regimes = result
elif analysis_type == "Multichannel (if multi-col)":
    from havolib.multichannel import MultichannelHAVOK
    X2 = np.column_stack([x, np.roll(x, 50), np.gradient(x), x**0.5])
    mh = MultichannelHAVOK(4, tau=1, m=20, r=3)
    mr = mh.fit_transform(X2, show_progress=False)
    forcing, risk = mr.joint_forcing, mr.joint_risk
    regimes = mr
else:
    from havolib.estimator import HavokEstimator
    from havolib.attribution import explain_forcing_spike
    est = HavokEstimator(m=30, r=5).fit(x)
    forcing, risk = est.forcing_, est.risk_
    spike_idx = np.argmax(np.abs(forcing))
    attr = explain_forcing_spike(x, spike_idx, m=20, r=3)
    regimes = None
    st.info(f"🔍 {attr['cause']}")

elapsed = (time.perf_counter() - t0) * 1000

# ── Plot ──
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    subplot_titles=("Raw Signal + Regimes", "Forcing", "Risk"))
fig.add_trace(go.Scatter(x=t, y=x, mode='lines', name='Signal', line=dict(color='#1f77b4',width=1)), row=1, col=1)
if regimes and hasattr(regimes, 'transition_points'):
    for tp in regimes.transition_points:
        fig.add_vline(x=t[tp] if tp<len(t) else t[-1], line_dash='dash', line_color='red',
                      annotation_text=f'Regime shift', row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=forcing, mode='lines', name='Forcing', line=dict(color='#d62728',width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=risk.astype(float), fill='tozeroy', name='Risk', line=dict(color='#ff7f0e')), row=3, col=1)
fig.update_layout(height=650, hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)

# ── Metrics ──
c1,c2,c3,c4 = st.columns(4)
c1.metric("Time", f"{elapsed:.0f} ms")
c2.metric("Max |f|", f"{np.max(np.abs(forcing)):.3f}")
c3.metric("Risk events", int(np.sum(risk)))
if regimes:
    if hasattr(regimes, 'segments'):
        c4.metric("Regimes", len(regimes.segments))
    elif hasattr(regimes, 'n_channels'):
        c4.metric("Channels", regimes.n_channels)

if regimes and hasattr(regimes, 'summary'):
    with st.expander("📋 Full Report"):
        st.text(regimes.summary())
