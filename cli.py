import click
import numpy as np

from havolib.data_loader import load_csv, generate_lorenz
from havolib.pipeline import HavokPipeline
from havolib.visualization import plot_dashboard

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
# === NEW: Pre-processing (deeper layer) ===
@click.option('--preprocess', is_flag=True, default=False, help='Enable full pre-processing (interpolate + outliers + smooth).')
@click.option('--smooth', default='savgol', type=click.Choice(['savgol', 'lowpass', 'none']), help='Smoothing method.')
@click.option('--smooth-window', default=11, help='Smoothing window size.')
@click.option('--outlier', default='iqr', type=click.Choice(['iqr', 'zscore', 'none']), help='Outlier removal method.')
# === NEW: Surrogate validation ===
@click.option('--surrogates', default=0, type=int, help='Number of phase-randomized surrogates for statistical validation (0 = disabled).')
def analyze(filepath, column, tau, m, auto, r, threshold_std, window, output,
            preprocess, smooth, smooth_window, outlier, surrogates):
    """Run HAVOK analysis on a CSV file and produce an interactive report."""
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


if __name__ == '__main__':
    cli()
