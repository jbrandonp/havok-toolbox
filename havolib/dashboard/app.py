import streamlit as st
import numpy as np
import plotly.io as pio

from havolib.data_loader import load_csv, generate_lorenz
from havolib.pipeline import HavokPipeline
from havolib.visualization import plot_dashboard

st.set_page_config(page_title="HAVOK Regime-Shift Detector", layout="wide")
st.title("🌪️ HAVOK Regime-Shift Detector")
st.markdown("Turn chaos into early-warning signals. Based on Brunton et al. (2017) HAVOK.")

with st.sidebar:
    st.header("Data Source")
    mode = st.radio("Choose input", ["Lorenz Demo", "Upload CSV"])

    if mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        column = st.text_input("Time series column name", value="value")
        if uploaded:
            try:
                data = load_csv(uploaded, column)
                t = np.arange(len(data))
            except Exception as e:
                st.error(str(e))
                st.stop()
        else:
            st.info("Upload a CSV to begin.")
            st.stop()
    else:
        n_points = st.slider("Lorenz points", 2000, 30000, 12000, step=1000)
        t, data = generate_lorenz(n_points=n_points)
        st.caption("Using Lorenz x-component (classic chaotic system)")

    st.header("HAVOK Parameters")
    tau = st.slider("Time delay τ", 1, 30, 1)
    m = st.slider("Embedding dimension m", 10, 300, 50, step=5)
    r = st.slider("Rank r (eigen-coordinates)", 2, 20, 5)
    threshold_std = st.slider("Risk threshold (std)", 1.0, 8.0, 3.0, 0.1)
    window = st.slider("Rolling window", 10, 500, 100, step=10)

    st.header("Deeper Layer Options")
    do_preprocess = st.checkbox("Enable pre-processing (interpolate/outliers/smooth)", value=False)
    run_surrogates = st.checkbox("Run surrogate validation", value=False)
    n_surrogates = st.slider("Number of surrogates", 10, 100, 30, step=10) if run_surrogates else 0
    run_button = st.button("🚀 Run HAVOK", type="primary")

if run_button:
    with st.spinner("Computing HAVOK embedding + forcing..."):
        pipeline = HavokPipeline(tau=tau, m=m, r=r,
                                 threshold_std=threshold_std, window=window)
        pipeline.fit(t, data)
        summary = None
        if n_surrogates > 0:
            with st.spinner("Running phase-randomized surrogates..."):
                summary = pipeline.validate_with_surrogates(n_surrogates=n_surrogates)
            pv = summary["p_value"]
            sa = summary["significant_at_alpha"]
            th = summary["surrogate_99th_percentile"]
            st.info(f"Surrogate p={pv:.3f} | significant={sa} | 99% thresh={th:.4f}")

    st.success("Analysis complete.")

    fig = plot_dashboard(
        pipeline.t_, pipeline.x_,
        pipeline.get_forcing(), pipeline.get_risk(),
        V=pipeline.get_eigen_coordinates()
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max |Forcing|", f"{np.max(np.abs(pipeline.get_forcing())):.4f}")
    col2.metric("Risk Events", int(np.sum(pipeline.get_risk())))
    col3.metric("Data Points", len(pipeline.t_))
    col4.metric("r (modes)", r)

    # Download
    html_bytes = pio.to_html(fig, include_plotlyjs=True).encode()
    st.download_button(
        "📥 Download interactive HTML report",
        data=html_bytes,
        file_name="havok_report.html",
        mime="text/html"
    )

    # Raw forcing data
    with st.expander("Show forcing signal data"):
        st.dataframe({
            "time": pipeline.t_,
            "forcing": pipeline.get_forcing(),
            "risk": pipeline.get_risk()
        })
else:
    st.info("Adjust parameters in the sidebar and click **Run HAVOK** to analyze.")
    st.markdown("""
    **How it works (quick):**
    1. Time-delay embedding (Hankel matrix)
    2. SVD → eigen-time-delay coordinates
    3. Isolate the intermittent forcing signal
    4. Threshold on forcing amplitude → regime-shift risk
    """)
