"""Generate sbatch files for use with SLURM using a Jinja2 template."""

import argparse
from pathlib import Path

import jinja2

TEMPLATE_PATH = Path(__file__).parent / "template.sbatch.jinja2"

DATASETS = [
    "cifar100",
    "core50",
    "imagenetr",
]

METHODS = [
    "ball",
    "clora",
    "ewc",
    "joint",
    "lora",
    "mas",
    "rwalk",
    "sdlora",
    "si",
    "tball",
]

N_RUNS = 5


def render_sbatch(
    dataset: str,
    method: str,
    n_runs: int,
    log_dir: str,
    time: str,
    gpus: int,
    cpus: int,
    mem: str,
    partition: str | None,
    account: str | None,
    conda_env: str | None,
    modules: list[str],
) -> str:
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATE_PATH.parent)),
        keep_trailing_newline=True,
    )
    template = jinja_env.get_template(TEMPLATE_PATH.name)
    return template.render(
        job_name=f"{dataset}_{method}",
        log_dir=log_dir,
        time=time,
        gpus=gpus,
        cpus=cpus,
        mem=mem,
        array=f"0-{n_runs - 1}",
        partition=partition,
        account=account,
        conda_env=conda_env,
        modules=modules,
        dataset=dataset,
        method=method,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate sbatch files for SLURM using a Jinja2 template."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        nargs="+",
        default=DATASETS,
        help="Dataset(s) to generate sbatch files for (default: all).",
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        nargs="+",
        default=METHODS,
        help="Method(s) to generate sbatch files for (default: all).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=N_RUNS,
        help="Number of runs (SLURM array size) (default: %(default)s).",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for SLURM output/error logs (default: %(default)s).",
    )
    parser.add_argument(
        "--time",
        default="24:00:00",
        help="Wall-clock time limit (default: %(default)s).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs per job (default: %(default)s).",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="Number of CPU cores per task (default: %(default)s).",
    )
    parser.add_argument(
        "--mem",
        default="32G",
        help="Memory per job (default: %(default)s).",
    )
    parser.add_argument(
        "--partition",
        default=None,
        help="SLURM partition to submit to.",
    )
    parser.add_argument(
        "--account",
        default=None,
        help="SLURM account to charge.",
    )
    parser.add_argument(
        "--conda-env",
        default=None,
        help="Conda environment to activate before running.",
    )
    parser.add_argument(
        "--module",
        dest="modules",
        action="append",
        default=[],
        metavar="MODULE",
        help="Environment module to load (can be repeated).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write sbatch files to. "
            "If omitted, scripts are printed to stdout."
        ),
    )
    args = parser.parse_args()

    for dataset in args.dataset:
        for method in args.method:
            content = render_sbatch(
                dataset=dataset,
                method=method,
                n_runs=args.n_runs,
                log_dir=args.log_dir,
                time=args.time,
                gpus=args.gpus,
                cpus=args.cpus,
                mem=args.mem,
                partition=args.partition,
                account=args.account,
                conda_env=args.conda_env,
                modules=args.modules,
            )

            if args.output_dir:
                out_path = Path(args.output_dir) / f"{dataset}_{method}.sbatch"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(content)
                print(f"Wrote {out_path}")
            else:
                print(content)


if __name__ == "__main__":
    main()
