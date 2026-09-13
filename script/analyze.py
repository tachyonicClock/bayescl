#!/usr/bin/env -S uv run
"""Statistical analysis of ``test``-stage results.

    ./script/analyze.py power [--delta 0.02] [--power 0.8] [RUNS]
    ./script/analyze.py compare [RUNS]

``power`` reads the ``pilot`` scale's ``test`` runs and reports how many
``full``-scale test seeds are needed to reliably detect an absolute
difference of ``--delta`` in any compared endpoint.

``compare`` reads the ``full`` scale's ``test`` runs and reports every
Mann-Whitney comparison between our methods and the baselines, Holm-Bonferroni
corrected across the full family of comparisons.
"""

import sys
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bayescl.analysis import required_test_runs, run_comparisons  # noqa: E402


@click.group()
def cli() -> None:
    """Analyze `test`-stage run results."""


@cli.command()
@click.argument(
    "runs", type=click.Path(exists=True, file_okay=False), default="./runs"
)
@click.option("--delta", type=float, default=0.02, show_default=True)
@click.option("--power", "target_power", type=float, default=0.8, show_default=True)
def power(runs: str, delta: float, target_power: float) -> None:
    """Required full-scale test seeds, sized from the pilot's variance."""
    n = required_test_runs(Path(runs), delta=delta, power=target_power)
    click.echo(f"Required test seeds (full scale): {n}")


@cli.command()
@click.argument(
    "runs", type=click.Path(exists=True, file_okay=False), default="./runs"
)
@click.option("--scale", type=click.Choice(["pilot", "full"]), default="full")
def compare(runs: str, scale: str) -> None:
    """Mann-Whitney + Holm-Bonferroni comparisons across all treatments."""
    comparisons = run_comparisons(Path(runs), scale=scale)
    df = pd.DataFrame([vars(c) for c in comparisons])
    with pd.option_context("display.max_rows", None, "display.width", None):
        click.echo(df.to_string(index=False))
    click.echo(f"\n{df['significant'].sum()} / {len(df)} comparisons significant")


if __name__ == "__main__":
    cli()
