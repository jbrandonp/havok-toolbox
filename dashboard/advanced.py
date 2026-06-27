"""
Advanced HAVOK Dashboard — multichannel comparison + what-if simulation.

Usage:
    streamlit run dashboard/advanced.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(page_title="HAVOK Pro Dashboard", layout="wide")

st.title("⚡ HAVOK Pro — Advanced Dashboard")
st.caption("Multichannel analysis · Method comparison · What-if simulation")

# ── Sidebar: Data + Parameters ────────────────────────────────

with st.sidebar:
    st.header("📊 Data Source")
    data_mode = st.radio("Mode", ["Synthetic (Lorenz)", "Synthetic (Sine+Jump)", "Upload CSV"],
                         index=0)

    n_points = st.slider("Points", 500, 5000, 2000)

    st.header("⚙️ HAVOK Parameters")
    tau = st.slider("τ (tau)", 1, 30, 1)
    m = st.slider("m (embedding)", 10, 100, 50)
    r = st.slider("r (modes)", 2, 15, 5)
    threshold = st.slider("Threshold σ", 1.0, 5.0, 3.0, 0.1)
    window = st.slider("Window", 20, 300, 100)

    st.header("🔬 What-If")
    intervention_amp = st.slider("Intervention amplitude", 0.0, 5.0, 2.0, 0.1,
                                 help="Simulate adding a control signal at the midpoint")

    run_btn = st.button("▶️ Run Analysis", type="primary", use_container_width=True)

# ── Main content ───────────────────────────────────────────────

if run_btn:
    with st.spinner("Running HAVOK analysis..."):
        # Generate data
        if data_mode == "Synthetic (Lorenz)":
            from havolib.data_loader import generate_lorenz
            t, x = generate_lorenz(n_points=n_points)
            source_label = "Lorenz Attractor"
        elif data_mode == "Synthetic (Sine+Jump)":
            t = np.linspace(0, 30 * np.pi, n_points)
            x = np.sin(t)
            x[n_points // 2:] *= 5
            x += np.random.randn(n_points) * 0.3
            source_label = "Sine + Amplitude Jump"
        else:
            st.info("Upload CSV to analyze")
            st.stop()

        # ---- HAVOK Analysis ----
        from havolib.estimator import HavokEstimator
        t0 = time.perf_counter()
        est = HavokEstimator(tau=tau, m=m, r=r, threshold_std=threshold, window=window)
        forcing = est.fit_transform(x)
        risk = est.risk_
        havok_time = (time.perf_counter() - t0) * 1000

        # ---- Baselines ----
        from benchmark.baselines import rolling_std_detector, cusum_detector
        t0 = time.perf_counter()
        baseline_risk = rolling_std_detector(x, window=window, n_std=threshold)
        baseline_time = (time.perf_counter() - t0) * 1000

        # ---- Edge of Chaos ----
        from havolib.edge_of_chaos import edge_of_chaos_score
        eoc = edge_of_chaos_score(x, tau=tau, m=m)

        # ---- What-If Simulation ----
        x_whatif = x.copy()
        mid = n_points // 2
        intervention = np.zeros(n_points)
        intervention[mid:mid + 100] = intervention_amp * np.sin(np.linspace(0, 4 * np.pi, 100))
        x_whatif[mid:] -= intervention[mid:]  # dampen the jump
        est_wi = HavokEstimator(tau=tau, m=m, r=r, threshold_std=threshold, window=window)
        forcing_wi = est_wi.fit_transform(x_whatif)

    # ── Row 1: Time Series ─────────────────────────────────────
    st.subheader(f"📈 Time Series — {source_label}")
    fig1 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         vertical_spacing=0.04,
                         subplot_titles=("Original + Intervention", "HAVOK Forcing", "Risk Comparison"))

    fig1.add_trace(go.Scatter(x=t, y=x, mode='lines', name='Original',
                              line=dict(color='#1f77b4', width=1)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=t, y=x_whatif, mode='lines', name='With Intervention',
                              line=dict(color='#2ca02c', width=1, dash='dash')), row=1, col=1)
    fig1.add_trace(go.Scatter(x=t, y=intervention, mode='lines', name='Intervention',
                              line=dict(color='#ff7f0e', width=2)), row=1, col=1)

    fig1.add_trace(go.Scatter(x=t, y=forcing, mode='lines', name='Forcing (original)',
                              line=dict(color='#d62728', width=1)), row=2, col=1)
    fig1.add_trace(go.Scatter(x=t, y=forcing_wi, mode='lines', name='Forcing (with intervention)',
                              line=dict(color='#2ca02c', width=1, dash='dot')), row=2, col=1)

    fig1.add_trace(go.Scatter(x=t, y=risk.astype(float), fill='tozeroy', name='HAVOK Risk',
                              line=dict(color='#d62728')), row=3, col=1)
    fig1.add_trace(go.Scatter(x=t, y=baseline_risk, fill='tozeroy', name='Rolling Std',
                              line=dict(color='#9467bd')), row=3, col=1)

    fig1.update_layout(height=700, hovermode='x unified')
    st.plotly_chart(fig1, use_container_width=True)

    # ── Row 2: Metrics + Edge of Chaos ─────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("HAVOK Time", f"{havok_time:.0f} ms")
    col2.metric("Baseline Time", f"{baseline_time:.0f} ms")
    col3.metric("Speedup", f"{baseline_time/havok_time:.1f}x" if havok_time > 0 else "N/A")
    col4.metric("Risk Events", f"{int(np.sum(risk))}")

    st.subheader("🌊 Edge of Chaos")
    c1, c2, c3 = st.columns(3)
    c1.metric("Lyapunov Exponent", f"{eoc['largest_lyapunov_exponent']:+.4f}")
    c2.metric("Edge Score", f"{eoc['edge_of_chaos_score']:.3f}")
    c3.metric("Interpretation", eoc['interpretation'])

    # ── Row 3: What-If Impact ──────────────────────────────────
    st.subheader("🎮 What-If: Impact of Intervention")
    forcing_before = np.max(np.abs(forcing[mid:]))
    forcing_after = np.max(np.abs(forcing_wi[mid:]))
    reduction = (forcing_before - forcing_after) / max(forcing_before, 1e-12) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Max |forcing| before", f"{forcing_before:.4f}")
    c2.metric("Max |forcing| after", f"{forcing_after:.4f}",
              delta=f"{-reduction:.1f}%" if reduction > 0 else f"+{-reduction:.1f}%")
    c3.metric("Risk reduction", f"{reduction:.1f}%")

    if reduction > 30:
        st.success(f"✅ Intervention effective — forcing reduced by {reduction:.1f}%")
    elif reduction > 10:
        st.info(f"ℹ️ Moderate reduction — forcing down {reduction:.1f}%")
    else:
        st.warning(f"⚠️ Minimal impact — only {reduction:.1f}% reduction")

    st.caption(f"Analysis completed in {(time.perf_counter() - t0)*1000:.0f} ms")
