"""CLI entry point for benchmark — also exposed as `havok benchmark`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
from benchmark.runner import run_benchmark, print_summary
from benchmark import ALL_DATASETS
from benchmark.baselines import BASELINES


@click.command("benchmark")
@click.option("--datasets", "-d", multiple=True, help="Dataset names to include (repeatable).")
@click.option("--methods", "-m", multiple=True, help="Methods to include (repeatable).")
@click.option("--all", "run_all", is_flag=True, default=True, help="Run all datasets + methods (default).")
@click.option("--quiet", "-q", is_flag=True, help="Suppress per-dataset progress.")
def run(datasets, methods, run_all, quiet):
    """Run HAVOK benchmark against baseline methods on regime-shift datasets."""

    ds_list = list(datasets) if datasets else None
    m_list = list(methods) if methods else None

    if ds_list:
        invalid = [d for d in ds_list if d not in ALL_DATASETS]
        if invalid:
            click.echo(f"Unknown dataset(s): {invalid}")
            click.echo(f"Available: {list(ALL_DATASETS.keys())}")
            return

    if m_list:
        all_methods = ["havok"] + list(BASELINES.keys())
        invalid = [m for m in m_list if m not in all_methods]
        if invalid:
            click.echo(f"Unknown method(s): {invalid}")
            click.echo(f"Available: {all_methods}")
            return

    click.echo("🚀 HAVOK Benchmark — Regime-Shift Detection")
    click.echo(f"   Datasets: {ds_list or 'all'} | Methods: {m_list or 'all'}")

    results = run_benchmark(datasets=ds_list, methods=m_list, verbose=not quiet)
    print_summary(results)


if __name__ == "__main__":
    run()
