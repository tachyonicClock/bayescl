"""Collect run results into a CSV for analysis.

Walks ``runs/{stage}/{scale}/{dataset}/{method}/{runid}/results.jsonl`` and
flattens every record into one CSV row.
"""

import sys
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bayescl.runio import read_jsonl  # noqa: E402


@click.command()
@click.argument("runs_root", type=click.Path(exists=True, file_okay=False), default="./runs")
@click.argument("output", type=click.Path())
@click.option(
    "--stage",
    type=click.Choice(["tune", "test"]),
    default="test",
    show_default=True,
)
def cli(runs_root: str, output: str, stage: str):
    rows = []
    for jsonl in sorted(Path(runs_root).glob(f"{stage}/*/*/*/*/results.jsonl")):
        scale, dataset, method, runid = jsonl.parts[-5:-1]
        for rec in read_jsonl(jsonl):
            rows.append(
                {
                    "stage": stage,
                    "scale": scale,
                    "dataset": dataset,
                    "method": method,
                    "runid": runid,
                    **rec,
                }
            )

    if not rows:
        click.echo(f"No {stage} results found under {runs_root}.")
        return

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.json_normalize(rows)
    df.to_csv(out, index=False)
    click.echo(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    cli()
