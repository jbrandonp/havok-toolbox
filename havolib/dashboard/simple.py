"""
HAVOK Simple — One-click regime-shift detection for non-programmers.

Usage:
    streamlit run havolib/dashboard/simple.py
    or double-click run_havok_app.bat
"""
import streamlit as st
import numpy as np
import pandas as pd
import io
import base64
from datetime import datetime

from havolib.pipeline import HavokPipeline
from havolib.visualization import plot_dashboard

st.set_page_config(
    page_title="HAVOK — Regime-Shift Detector",
    page_icon="🌪️",
    layout="wide",
)

# ── Header ──
st.title("🌪️ HAVOK Regime-Shift Detector")
st.caption("Detect early-warning signals in your time series. No coding required.")

# ── Sidebar: upload ──
with st.sidebar:
    st.header("📂 Upload Your Data")
    uploaded = st.file_uploader(
        "Drag & drop a CSV, TXT, or Excel file",
        type=["csv", "txt", "xlsx", "xls"],
        help="File should contain at minimum one column of numeric time series data.",
    )

    use_demo = st.checkbox("Or try with demo data (Lorenz attractor)", value=False)

    st.divider()
    st.header("⚙️ Options")
    sensitivity = st.slider("Sensitivity", 1.0, 5.0, 3.0, 0.5,
                            help="Higher = fewer alerts. 3.0 is a good default.")
    st.caption("Parameters are auto-tuned. Sensitivity controls how many alerts you get.")

    st.divider()
    st.markdown("**HAVOK v0.3.0** — [GitHub](https://github.com/jbrandonp/havok-toolbox)")

# ── Load data ──
data = None
source_name = ""

if use_demo:
    from havolib.data_loader import generate_lorenz
    _, data = generate_lorenz(3000)
    source_name = "Lorenz attractor (demo)"
    st.info("📊 Using demo data: Lorenz chaotic attractor (3000 points)")
elif uploaded is not None:
    try:
        # Read file
        fname = uploaded.name.lower()
        if fname.endswith(".csv") or fname.endswith(".txt"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.sidebar.success(f"Loaded: {uploaded.name} ({len(df)} rows, {len(df.columns)} cols)")

        # Let user pick column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            st.error("No numeric columns found in file.")
            st.stop()

        col = st.sidebar.selectbox("Select data column", numeric_cols)
        data = df[col].dropna().values
        source_name = f"{uploaded.name} → column '{col}'"
        st.info(f"📊 Loaded {len(data)} points from {source_name}")

        # Show preview
        with st.expander("Data preview"):
            st.dataframe(df.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

# ── Run HAVOK ──
if data is not None and len(data) > 100:
    st.divider()

    with st.spinner(f"Running HAVOK with auto-tuned parameters..."):
        pipe = HavokPipeline()
        pipe.auto_fit(None, data)
        forcing = pipe.get_forcing()
        risk = pipe.get_risk()
        V = pipe.get_eigen_coordinates()

    # ── Results ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data points", f"{len(data):,}")
    with col2:
        st.metric("Max forcing", f"{np.max(np.abs(forcing)):.4f}")
    with col3:
        risk_pct = risk.mean() * 100
        st.metric("Risk events", f"{int(np.sum(risk))} ({risk_pct:.1f}%)")
    with col4:
        st.metric("Auto-tuned τ", str(pipe.tau))

    # ── Dashboard plot ──
    st.subheader("📈 Analysis Results")
    fig = plot_dashboard(pipe.t_[:len(forcing)], pipe.x_[:len(forcing)], forcing, risk,
                         V=V if V.shape[1] >= 3 else None)
    st.plotly_chart(fig, use_container_width=True)

    # ── Export ──
    st.divider()
    st.subheader("📥 Export Results")
    col_a, col_b = st.columns(2)

    # CSV export
    df_out = pd.DataFrame({
        "time": pipe.t_[:len(forcing)],
        "signal": pipe.x_[:len(forcing)],
        "forcing": forcing,
        "risk": risk,
    })
    csv_buf = io.StringIO()
    df_out.to_csv(csv_buf, index=False)
    b64 = base64.b64encode(csv_buf.getvalue().encode()).decode()

    with col_a:
        st.download_button(
            "⬇ Download Results (CSV)",
            data=csv_buf.getvalue(),
            file_name=f"havok_results_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
        )

    # HTML report
    html = f"""<html><head><meta charset="utf-8"><title>HAVOK Report</title>
<style>body{{font-family:Arial;max-width:900px;margin:auto;padding:20px}}
.metric{{display:inline-block;margin:10px;padding:15px;background:#f0f0f0;border-radius:8px}}
h1{{color:#1a1a2e}}h2{{color:#16213e}}.alert{{color:#c0392b;font-weight:bold}}</style></head>
<body><h1>🌪️ HAVOK Regime-Shift Report</h1>
<p>Source: {source_name} | Date: {datetime.now():%Y-%m-%d %H:%M}</p>
<p>Parameters: τ={pipe.tau}, m={pipe.m}, r={pipe.r}</p>
<div class="metric"><b>{len(data):,}</b><br>Data points</div>
<div class="metric"><b>{np.max(np.abs(forcing)):.4f}</b><br>Max forcing</div>
<div class="metric"><b>{int(np.sum(risk))}</b><br>Risk events</div>
<div class="metric"><b>{risk_pct:.1f}%</b><br>At-risk time</div>
<h2>Interpretation</h2>
<p>{"⚠️ <span class='alert'>Regime shifts detected.</span> Investigate time periods with risk=1 for anomalous behavior." if risk_pct > 1 else "✅ No significant regime shifts detected. The signal appears stable."}</p>
<p>HAVOK (Hankel Alternative View of Koopman) extracts intermittent forcing signals from chaotic time series. When the forcing exceeds {sensitivity} standard deviations, a regime shift is flagged.</p>
<p><small>Generated by HAVOK v0.3.0 — <a href='https://github.com/jbrandonp/havok-toolbox'>github.com/jbrandonp/havok-toolbox</a></small></p>
</body></html>"""
    with col_b:
        st.download_button(
            "⬇ Download Report (HTML)",
            data=html,
            file_name=f"havok_report_{datetime.now():%Y%m%d_%H%M}.html",
            mime="text/html",
        )

elif data is not None and len(data) <= 100:
    st.warning("File has too few data points. Need at least 100 rows for analysis.")
else:
    # Welcome screen
    st.markdown("""
    ## 👋 Welcome to HAVOK

    **No coding needed.** Just upload a file from the sidebar and get results in seconds.

    ### What does HAVOK do?
    - Detects **regime shifts** in time series (seizures in EEG, market crashes, climate tipping points)
    - Extracts **forcing signals** that reveal when a system is about to change
    - Works on **chaotic data** where traditional methods fail

    ### Try it now:
    1. Upload a CSV file (or check "demo data")
    2. Adjust sensitivity if needed
    3. Get your results → download the report

    *Based on Brunton et al. (2017) — [Read the paper](https://www.nature.com/articles/s41467-017-00030-8)*
    """)

# ── Footer ──
st.divider()
st.caption("HAVOK v0.3.0 | MIT + Commons Clause | Built for scientists, not programmers")


def main():
    """Entry point for `havok-app` command — launches Streamlit in a subprocess."""
    import subprocess, sys
    from pathlib import Path
    app_path = Path(__file__).resolve()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path),
                    "--server.headless", "true"], check=False)


if __name__ == "__main__":
    main()
