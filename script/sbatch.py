import os

import click
import jinja2

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.dirname(__file__)),
    autoescape=jinja2.select_autoescape(),
)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("dataset", required=True)
@click.argument("method", required=True)
@click.option(
    "--duration",
    help="Duration for the job in format HH:MM:SS.",
    default="12:00:00",
)
@click.option(
    "--memory",
    help="Memory for the job in GB.",
    default=None,
    type=int,
)
def hp(dataset, method, duration, memory):
    template = env.get_template("hp.sbatch.jinja2")
    rendered = template.render(
        dataset=dataset, method=method, duration=duration, memory=memory
    )
    print(rendered)


@cli.command()
@click.argument("dataset", required=True)
@click.argument("method", required=True)
@click.option(
    "--duration",
    help="Duration for the job in format HH:MM:SS.",
    default="01:30:00",
)
@click.option(
    "--memory",
    help="Memory for the job in GB.",
    default=None,
    type=int,
)
@click.option(
    "--n-runs",
    help="Number of runs for the evaluation.",
    default=5,
    type=int,
)
def eval(dataset, method, duration, memory, n_runs):
    template = env.get_template("eval.sbatch.jinja2")
    rendered = template.render(
        dataset=dataset, method=method, duration=duration, memory=memory, n_runs=n_runs
    )
    print(rendered)


if __name__ == "__main__":
    cli()
