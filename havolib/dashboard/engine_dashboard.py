"""
HAVOK Engine Live Dashboard — Streamlit app for multi-stream monitoring.

Usage:
    streamlit run dashboard/engine_dashboard.py
"""
import streamlit as st
import numpy as np
import time
import threading
from collections import deque

st.set_page_config(page_title="HAVOK Engine Dashboard", layout="wide")
st.title("🌪️ HAVOK Engine — Live Monitor")

# Sidebar
with st.sidebar:
    st.header("Stream Configuration")
    num_streams = st.slider("Number of streams", 1, 4, 2)
    stream_configs = []
    for i in range(num_streams):
        with st.expander(f"Stream {i+1}", expanded=(i == 0)):
            stype = st.selectbox("Type", ["lorenz", "eeg"], key=f"type_{i}")
            m = st.slider("m (embedding)", 10, 200, 50, key=f"m_{i}")
            tau = st.slider("tau (delay)", 1, 30, 1 if stype == "lorenz" else 5, key=f"tau_{i}")
            r = st.slider("r (rank)", 2, 15, 5, key=f"r_{i}")
            stream_configs.append({"type": stype, "m": m, "tau": tau, "r": r})

    run_btn = st.button("▶️ Start Engine", type="primary")
    stop_btn = st.button("⏹️ Stop")

# Main content
if "engine_running" not in st.session_state:
    st.session_state.engine_running = False
    st.session_state.data = {}
    st.session_state.risk_history = {}
    st.session_state.alert_log = deque(maxlen=50)

if run_btn:
    st.session_state.engine_running = True
    st.session_state.data = {i: deque(maxlen=5000) for i in range(num_streams)}
    st.session_state.forcing = {i: deque(maxlen=5000) for i in range(num_streams)}
    st.session_state.risk = {i: deque(maxlen=5000) for i in range(num_streams)}

if stop_btn:
    st.session_state.engine_running = False

# Layout
cols = st.columns(min(num_streams, 3))
chart_cols = st.columns(min(num_streams, 3))

if st.session_state.engine_running:
    from havolib.engine.incremental_havok import IncrementalHAVOK
    from havolib.engine.risk_engine import RiskEngine
    from havolib.data_loader import generate_lorenz, generate_eeg_like

    risk_engine = RiskEngine()

    # Generate and process data
    generators = []
    havoks = []
    for i, cfg in enumerate(stream_configs):
        if cfg["type"] == "lorenz":
            _, data = generate_lorenz(n_points=5000, dt=0.01)
            generators.append(iter(data))
        else:
            _, data = generate_eeg_like(n_points=5000)
            generators.append(iter(data))
        havoks.append(IncrementalHAVOK(m=cfg["m"], tau=cfg["tau"], r=cfg["r"]))

    # Process batches
    batch_size = 50
    for _ in range(min(5000 // batch_size, 80)):
        if not st.session_state.engine_running:
            break
        for i, (gen, havok) in enumerate(zip(generators, havoks)):
            for _ in range(batch_size):
                try:
                    val = next(gen)
                    forcing, risk = havok.update(float(val))
                    st.session_state.data[i].append(val)
                    st.session_state.forcing[i].append(forcing)
                    st.session_state.risk[i].append(risk)
                except StopIteration:
                    pass

        # Risk assessment every batch
        for i, havok in enumerate(havoks):
            if havok.point_count > 100:
                fhist = havok.get_forcing_history(200)
                score, level, details = risk_engine.assess(fhist)
                if level.value in ("warning", "critical"):
                    st.session_state.alert_log.appendleft({
                        "time": time.strftime("%H:%M:%S"),
                        "stream": f"Stream {i+1}",
                        "level": level.value.upper(),
                        "risk": f"{score:.2f}",
                        "details": f"surge={details['surge']:.2f} trend={details['trend']:.2f}",
                    })

    st.session_state.engine_running = False
    st.success("Processing complete — review results below.")
    st.rerun()

# Display
for i in range(num_streams):
    with cols[i % len(cols)]:
        if i in st.session_state.get("risk", {}):
            hist = list(st.session_state.risk[i])
            if hist:
                current_risk = hist[-1] if hist else 0
                color = "🔴" if current_risk > 0.7 else "🟠" if current_risk > 0.4 else "🟢"
                st.metric(f"Stream {i+1}", f"{color} {current_risk:.3f}",
                          delta=f"{len(hist)} pts")

    if i in st.session_state.get("data", {}) and len(st.session_state.data[i]) > 0:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=list(st.session_state.forcing[i])[-500:],
            mode='lines', name=f'Forcing {i+1}',
            line=dict(color='#d62728', width=1)))
        fig.update_layout(height=200, margin=dict(t=5, b=5, l=5, r=5),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}")

# Alert log
if st.session_state.alert_log:
    st.subheader("🚨 Alert Log")
    alerts_data = list(st.session_state.alert_log)
    st.dataframe(alerts_data, use_container_width=True)
