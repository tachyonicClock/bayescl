"""Collect experiment results into CSV files for analysis."""

import os
import pickle
from pathlib import Path
from typing import Dict

import click
import optuna
import pandas as pd
import yaml

script_dir = Path(__file__).parent
const = yaml.safe_load((script_dir / "const.yml").open())


@click.group()
def cli():
    pass


@cli.command()
@click.argument("study")
@click.argument("output", type=click.Path())
def hpsearch(study: str, output: str):
    """Collect hyperparameter search trials from Optuna into OUTPUT CSV.

    STUDY is the short study identifier, e.g. 'example-study'.
    Loads studies named bayescl/STUDY/{dataset}/{method} for all datasets and methods.
    """
    output: Path = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    storage_url = os.environ.get("OPTUNA_STORAGE")
    if not storage_url:
        raise click.ClickException("OPTUNA_STORAGE is not set.")

    storage = optuna.storages.RDBStorage(storage_url)
    study_dfs = []

    for dataset in const["datasets"]:
        for method in const["methods"]:
            study_name = f"bayescl/{study}/{dataset}/{method}"
            try:
                loaded = optuna.load_study(study_name=study_name, storage=storage)
            except KeyError:
                click.echo(f"Study '{study_name}' not found, skipping...")
                continue

            df: pd.DataFrame = loaded.trials_dataframe()
            study_dfs.append(df.assign(dataset=dataset, method=method))

    result_df = pd.concat(study_dfs, ignore_index=True)
    result_df.to_csv(output, index=False)
    click.echo(f"Wrote {len(result_df)} rows to {output}")


@cli.command()
@click.argument("study")
@click.argument("output", type=click.Path())
@click.option(
    "--log-root",
    default="log",
    show_default=True,
    help="Root directory for run logs.",
)
def run(study: str, output: str, log_root: str):
    """Collect evaluation run results from log files into OUTPUT CSV.

    STUDY is the run name used when invoking 'main.py run STUDY SEED'.
    Scans log/{log_root}/{study}/{dataset}/{method}/{run_id}/metrics.pkl
    """
    log_path = Path(log_root) / study
    if not log_path.exists():
        raise click.ClickException(f"Log directory '{log_path}' does not exist.")

    records = []

    for metrics_file in sorted(log_path.rglob("metrics.pkl")):
        parts = metrics_file.parts
        # Expected structure: log_root / study / dataset / method / run_id / metrics.pkl
        if len(parts) < 5:
            continue
        dataset = parts[-4]
        method = parts[-3]
        run_id = parts[-2]

        with open(metrics_file, "rb") as f:
            metrics: dict = pickle.load(f)

        record: Dict[str, str | int | float] = {
            "dataset": dataset,
            "method": method,
            "run_id": run_id,
        }
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                record[key] = value
        records.append(record)

    if not records:
        click.echo("No runs found.")
        return

    result_df = pd.DataFrame(records)
    result_df.to_csv(output, index=False)
    click.echo(f"Wrote {len(result_df)} rows to {output}")


if __name__ == "__main__":
    cli()
