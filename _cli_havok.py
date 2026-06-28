import click
import numpy as np
import yaml
from pathlib import Path

from havolib.data_loader import load_csv, generate_lorenz
from havolib.pipeline import HavokPipeline
from havolib.visualization import plot_dashboard
from havolib.ml_risk_predictor import quick_forcing_risk
from havolib.edge_of_chaos import edge_of_chaos_score

# Lazy import for optional viz dep (installed as part of base now)
def _get_pio():
    import plotly.io as pio
    return pio


@click.group()
@click.version_option()
def cli():
    """HAVOK Regime-Shift Detector CLI.

    The right way to choose tau and m is with Mutual Information + False Nearest Neighbors.
    Use --auto on analyze or the 'suggest' command.

    New deeper-layer features:
      --preprocess          Enable interpolation + outlier removal + smoothing
      --surrogates N        Run phase-randomized surrogate test after analysis

    Portable install:
      pip install -e .
      havok demo
      havok analyze data.csv -c value
    """
    pass


@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--column', '-c', required=True, help='Column name containing the time series.')
@click.option('--tau', default=None, show_default=True, help='Time delay (override auto).')
@click.option('--m', default=None, show_default=True, help='Embedding dim (override auto).')
@click.option('--auto', is_flag=True, default=True, help='Auto-tune tau/m using MI + FNN (recommended).')
@click.option('--r', default=5, show_default=True, help='Number of eigen-time-delay coordinates.')
@click.option('--threshold-std', default=3.0, show_default=True, help='Risk threshold in std devs.')
@click.option('--window', default=100, show_default=True, help='Rolling window for risk.')
@click.option('--output', '-o', default='havok_report.html', show_default=True,
              help='Output HTML file.')
@click.option('--config', '-C', 'profile', default=None, 
              help='Load params from havok_config.yaml profile (eeg, finance, climate, lorenz_demo).')
# === NEW: Pre-processing (deeper layer) ===
@click.option('--preprocess', is_flag=True, default=False, help='Enable full pre-processing (interpolate + outliers + smooth).')
@click.option('--smooth', default='savgol', type=click.Choice(['savgol', 'lowpass', 'none']), help='Smoothing method.')
@click.option('--smooth-window', default=11, help='Smoothing window size.')
@click.option('--outlier', default='iqr', type=click.Choice(['iqr', 'zscore', 'none']), help='Outlier removal method.')
# === NEW: Surrogate validation ===
@click.option('--surrogates', default=0, type=int, help='Number of phase-randomized surrogates for statistical validation (0 = disabled).')
def analyze(filepath, column, tau, m, auto, r, threshold_std, window, output, profile,
            preprocess, smooth, smooth_window, outlier, surrogates):
    """Run HAVOK analysis on a CSV file and produce an interactive report."""
    from havolib.config import load_config as load_havok_config

    # Load profile if specified
    if profile:
        cfg = load_havok_config(profile=profile)
        if tau is None:
            tau = cfg["tau"]
        if m is None:
            m = cfg["m"]
        r = cfg["r"]
        threshold_std = cfg["threshold_std"]
        window = cfg["window"]
        preproc = cfg.get("preprocess", {})
        if not preprocess:
            preprocess = any(preproc.values())
        click.echo(f"Loaded profile '{profile}': tau={tau}, m={m}, r={r}")

    click.echo(f"Loading {filepath} column '{column}'...")
    data = load_csv(filepath, column)
    t = np.arange(len(data))

    pipeline = HavokPipeline(
        r=r, threshold_std=threshold_std, window=window,
        do_preprocess=preprocess,
        smooth_method=None if smooth == 'none' else smooth,
        smooth_window=smooth_window,
        outlier_method=None if outlier == 'none' else outlier
    )

    if auto and (tau is None or m is None):
        click.echo("Auto-tuning tau and m using Mutual Information + False Nearest Neighbors...")
        params = pipeline.suggest_parameters(data)
        tau = params["tau"] if tau is None else tau
        m = params["m"] if m is None else m
        click.echo(f"  → Auto-selected: tau={tau}, m={m}")

    pipeline.tau = tau if tau is not None else 1
    pipeline.m = m if m is not None else 50

    pipeline.fit(t, data)

    # Optional surrogate validation (deeper layer)
    if surrogates > 0:
        click.echo(f"Running {surrogates} phase-randomized surrogates for statistical validation...")
        summary = pipeline.validate_with_surrogates(n_surrogates=surrogates)
        click.echo(f"  p-value: {summary['p_value']:.3f} | 99% surrogate threshold: {summary['surrogate_99th_percentile']:.4f}")
        click.echo(f"  Significant: {summary['significant_at_alpha']}")

    fig = plot_dashboard(pipeline.t_, pipeline.x_,
                         pipeline.get_forcing(), pipeline.get_risk(),
                         V=pipeline.get_eigen_coordinates())

    _get_pio().write_html(fig, file=output, auto_open=False)
    click.echo(f"✅ Report saved to {output}")
    click.echo(f"   Max |forcing|: {np.max(np.abs(pipeline.get_forcing())):.4f}")
    click.echo(f"   Risk events: {int(np.sum(pipeline.get_risk()))}")


@cli.command()
@click.option('--n', default=12000, show_default=True, help='Number of points for Lorenz system.')
@click.option('--preprocess', is_flag=True, default=False, help='Run with pre-processing enabled.')
@click.option('--surrogates', default=0, type=int, help='Run surrogate validation with N surrogates.')
def demo(n, preprocess, surrogates):
    """Run HAVOK on a simulated Lorenz attractor (best way to see forcing spikes)."""
    click.echo("Generating Lorenz attractor...")
    t, x = generate_lorenz(n_points=n)

    pipeline = HavokPipeline(do_preprocess=preprocess)
    click.echo("Auto-tuning on Lorenz data...")
    pipeline.auto_fit(t, x)

    if surrogates > 0:
        click.echo(f"Running {surrogates} phase-randomized surrogates...")
        summary = pipeline.validate_with_surrogates(n_surrogates=surrogates)
        click.echo(f"  p-value: {summary['p_value']:.3f} | significant: {summary['significant_at_alpha']}")

    fig = plot_dashboard(pipeline.t_, pipeline.x_,
                         pipeline.get_forcing(), pipeline.get_risk(),
                         V=pipeline.get_eigen_coordinates())

    output = "lorenz_demo.html"
    _get_pio().write_html(fig, file=output, auto_open=False)
    click.echo(f"✅ Lorenz demo report saved to {output}")
    click.echo("Open it in your browser to see the forcing spikes before chaotic bursts.")
    click.echo(f"Max |forcing| = {np.max(np.abs(pipeline.get_forcing())):.4f}")


@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--column', '-c', required=True, help='Column name.')
@click.option('--max-lag', default=100, show_default=True)
@click.option('--max-m', default=50, show_default=True)
def suggest(filepath, column, max_lag, max_m):
    """Suggest optimal tau and m for a dataset using MI + FNN."""
    data = load_csv(filepath, column)
    pipeline = HavokPipeline()
    params = pipeline.suggest_parameters(data, max_lag=max_lag, max_m=max_m)
    click.echo(f"Suggested parameters for {filepath} ({column}):")
    click.echo(f"  tau = {params['tau']}")
    click.echo(f"  m   = {params['m']}")
    click.echo(f"  method: {params['method']}")
    click.echo("Use these with: havok analyze ... --tau X --m Y")


@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--column', '-c', required=True, help='Column name.')
@click.option('--horizon', default=30, show_default=True, help='Prediction horizon.')
@click.option('--reservoir', default=150, show_default=True, help='ESN reservoir size.')
@click.option('--output', '-o', default='havok_predict_report.html', show_default=True)
def predict(filepath, column, horizon, reservoir, output):
    """Predict future forcing and regime-shift risk using Echo State Network.

    Requires 'havok analyze' to be run first, or provide a CSV.
    Runs HAVOK extraction then ESN prediction on the forcing signal.
    """
    click.echo(f"Loading {filepath} column '{column}'...")
    data = load_csv(filepath, column)
    t = np.arange(len(data))

    # Step 1: Extract forcing via HAVOK
    click.echo("Extracting forcing signal via HAVOK...")
    pipeline = HavokPipeline()
    pipeline.auto_fit(t, data)
    forcing = pipeline.get_forcing()

    click.echo(f"  Max |forcing| = {np.max(np.abs(forcing)):.4f}")

    # Step 2: ESN prediction
    click.echo(f"Training Echo State Network (N={reservoir})...")
    result = quick_forcing_risk(
        forcing,
        horizon=horizon,
        reservoir_size=reservoir,
    )

    risk_pct = result["regime_shift_risk"] * 100
    click.echo(f"  Regime-shift risk: {risk_pct:.1f}%")
    click.echo(f"  Predicted forcing (first 5): {result['predicted_forcing'][:5].round(4)}")

    # Step 3: Edge of chaos
    click.echo("Computing edge-of-chaos metrics...")
    eoc = edge_of_chaos_score(data)
    click.echo(f"  Lyapunov exponent: {eoc['largest_lyapunov_exponent']:.4f}")
    click.echo(f"  Critical slowing down: {eoc['critical_slowing_down_lag1']:.3f}")
    click.echo(f"  Edge-of-chaos score: {eoc['edge_of_chaos_score']:.2f}")
    click.echo(f"  → {eoc['interpretation']}")

    # Generate report
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Forcing Signal + Predicted", "Regime-Shift Risk", "Predicted Forcing (zoom)")
    )

    fig.add_trace(
        go.Scatter(y=forcing, mode='lines', name='Forcing (history)', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    future_x = np.arange(len(forcing), len(forcing) + horizon)
    fig.add_trace(
        go.Scatter(x=future_x, y=result['predicted_forcing'],
                   mode='lines+markers', name='Predicted', line=dict(color='#d62728', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=result['predicted_forcing'], mode='lines', name='Predicted', line=dict(color='#d62728')),
        row=3, col=1
    )
    fig.add_hline(y=0, line_dash='dash', line_color='gray', row=3, col=1)

    risk_array = np.zeros(horizon)
    risk_array[-1] = result['regime_shift_risk']
    fig.add_trace(
        go.Bar(y=[result['regime_shift_risk']], name='Risk', marker_color='#ff7f0e',
               text=f"{risk_pct:.1f}%", textposition='auto'),
        row=2, col=1
    )

    fig.update_layout(height=800, title_text="HAVOK ESN Predictor — Forcing Forecast & Risk")

    _get_pio().write_html(fig, file=output, auto_open=False)
    click.echo(f"✅ Predict report saved to {output}")


@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--column', '-c', required=True, help='Column name.')
def chaos(filepath, column):
    """Compute edge-of-chaos metrics for a time series."""
    data = load_csv(filepath, column)
    eoc = edge_of_chaos_score(data)

    click.echo(f"Edge-of-Chaos Analysis for {filepath} ({column}):")
    click.echo(f"  Lyapunov exponent:     {eoc['largest_lyapunov_exponent']:+.4f}")
    click.echo(f"  Critical slowing down: {eoc['critical_slowing_down_lag1']:.3f}")
    click.echo(f"  Edge-of-chaos score:   {eoc['edge_of_chaos_score']:.3f}")
    click.echo(f"  → {eoc['interpretation']}")


@cli.group()
def engine():
    """Start and manage the HAVOK streaming engine."""
    pass


@engine.command("start")
@click.option("--config", "-c", default="engine.yaml", show_default=True, help="Engine config file.")
@click.option("--duration", default=0, help="Run for N seconds (0 = forever).")
def engine_start(config, duration):
    """Start the HAVOK streaming engine."""
    import asyncio
    from havolib.engine.engine import HavokEngine

    click.echo(f"🚀 Starting HAVOK Engine with config: {config}")

    eng = HavokEngine(config_path=config)
    click.echo(f"   Streams: {list(eng._streams.keys())}")

    async def run_engine():
        await eng.start()
        if duration > 0:
            await asyncio.sleep(duration)
            await eng.stop()
        else:
            try:
                while True:
                    await asyncio.sleep(10)
                    states = eng.get_all_states()
                    for sid, state in states.items():
                        if state:
                            click.echo(f"  [{sid}] pts={state['points_processed']} forcing={state['latest_forcing']:.4f}")
            except KeyboardInterrupt:
                pass
            finally:
                await eng.stop()

    asyncio.run(run_engine())
    click.echo("Engine stopped.")


@engine.command("list")
def engine_list():
    """List available engine config streams."""
    import yaml
    try:
        with open("engine.yaml") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        click.echo("No engine.yaml found. Create one with 'havok engine init'.")
        return

    for s in cfg.get("streams", []):
        alerts = len(s.get("alerts", []))
        click.echo(f"  {s['id']:20s} → {s['source']:25s} ({alerts} alert rules)")


@engine.command("init")
def engine_init():
    """Create a default engine.yaml config."""
    from pathlib import Path
    import shutil
    src = Path(__file__).parent / "engine.yaml"
    dst = Path.cwd() / "engine.yaml"
    if dst.exists():
        click.echo(f"{dst} already exists. Delete it first or use --force.")
    else:
        shutil.copy(str(src), str(dst))
        click.echo(f"Created {dst}")


@cli.command("benchmark")
@click.option("--datasets", "-d", multiple=True, help="Dataset names (repeatable).")
@click.option("--methods", "-m", multiple=True, help="Method names (repeatable).")
@click.option("--quiet", "-q", is_flag=True, help="Suppress per-dataset output.")
def benchmark_cmd(datasets, methods, quiet):
    """Benchmark HAVOK vs baselines on regime-shift datasets."""
    from havolib.benchmark.runner import run_benchmark, print_summary
    ds = list(datasets) if datasets else None
    ms = list(methods) if methods else None
    click.echo("🚀 HAVOK Benchmark — Regime-Shift Detection")
    results = run_benchmark(datasets=ds, methods=ms, verbose=not quiet)
    print_summary(results)


@cli.command("analyze")
@click.argument("file", type=click.Path(exists=True), required=False)
@click.option("--column", "-c", default=None, help="Column name for the signal (auto-detect if omitted).")
@click.option("--output", "-o", default=None, help="Save results to CSV.")
@click.option("--report", "-r", default=None, help="Save HTML report.")
def analyze_cmd(file, column, output, report):
    """One-click analysis: havok analyze data.csv [--column price] [--output results.csv]

    No parameters needed — auto-tunes tau, m, r automatically.
    """
    import pandas as pd
    from havolib.pipeline import HavokPipeline

    if file is None:
        click.echo("Usage: havok analyze <file.csv> [--column NAME] [--output results.csv]")
        click.echo("Example: havok analyze eeg_data.csv --column Fp1")
        return

    # Load data
    fname = str(file).lower()
    if fname.endswith(".csv") or fname.endswith(".txt"):
        df = pd.read_csv(file)
    elif fname.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        click.echo(f"Unsupported file format: {file}")
        return

    # Detect or select column
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if column and column in df.columns:
        data = df[column].dropna().values
    elif len(numeric_cols) > 0:
        column = numeric_cols[0]
        data = df[column].dropna().values
        click.echo(f"Auto-detected column: '{column}' ({len(data)} points)")
    else:
        click.echo("No numeric columns found.")
        return

    if len(data) < 100:
        click.echo(f"Need at least 100 data points, got {len(data)}.")
        return

    # Auto-tune + run
    click.echo("Auto-tuning parameters...")
    pipe = HavokPipeline()
    pipe.auto_fit(None, data)
    forcing = pipe.get_forcing()
    risk = pipe.get_risk()

    max_f = np.max(np.abs(forcing))
    n_risk = int(np.sum(risk))
    risk_pct = float(np.mean(risk)) * 100

    click.echo(f"\nResults (τ={pipe.tau}, m={pipe.m}, r={pipe.r}):")
    click.echo(f"  Data points:  {len(data)}")
    click.echo(f"  Max forcing:  {max_f:.4f}")
    click.echo(f"  Risk events:  {n_risk} ({risk_pct:.1f}% of signal)")

    if risk_pct > 1:
        click.echo(f"  ⚠️  Regime shifts detected at {risk_pct:.1f}% of timesteps.")
    else:
        click.echo("  ✅ No significant regime shifts detected.")

    # Export
    if output:
        pd.DataFrame({"forcing": forcing, "risk": risk}).to_csv(output, index=False)
        click.echo(f"\nResults saved to {output}")

    if report:
        with open(report, "w") as f:
            f.write(f"""<html><body>
<h1>HAVOK Analysis: {file}</h1>
<p>Parameters: τ={pipe.tau}, m={pipe.m}, r={pipe.r}</p>
<p>Max forcing: {max_f:.4f} | Risk events: {n_risk} ({risk_pct:.1f}%)</p>
</body></html>""")
        click.echo(f"Report saved to {report}")


if __name__ == '__main__':
    cli()
