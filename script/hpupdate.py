from os import environ
from pathlib import Path

import click
import jinja2
import optuna
import yaml

MAX_TRIALS = 30

script_dir = Path(__file__).parent
const = yaml.safe_load((script_dir / "const.yml").open())


def score(acc: float, ece: float) -> float:
    return (acc + (1 - ece)) / 2


@click.command()
@click.option(
    "-s",
    "--study",
    "study_name",
    required=True,
    help="Short study identifier, e.g. 'example-study'.",
)
@click.option(
    "-d",
    "--dataset",
    "datasets",
    type=click.Choice(const["datasets"]),
    multiple=True,
    required=True,
    help="Dataset to update. Repeat to update multiple datasets.",
)
@click.option(
    "-m",
    "--method",
    "methods",
    type=click.Choice(const["methods"]),
    multiple=True,
    required=True,
    help="Method to update. Repeat to update multiple methods.",
)
@click.option(
    "--max-trials",
    default=MAX_TRIALS,
    show_default=True,
    type=int,
    help="Maximum number of completed/pruned trials to consider.",
)
def main(
    study_name: str,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    max_trials: int,
):
    storage_url = environ.get("OPTUNA_STORAGE")
    if not storage_url:
        raise click.ClickException("OPTUNA_STORAGE is not set.")

    storage = optuna.storages.RDBStorage(storage_url)
    jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
    jinja_template = jinja_env.get_template(
        (script_dir / "hpupdate.jsonnet.jinja2").as_posix()
    )

    studies = [
        f"bayescl/{study_name}/{dataset}/{method}"
        for dataset in datasets
        for method in methods
    ]

    for full_study_name in studies:
        try:
            study = optuna.load_study(study_name=full_study_name, storage=storage)
        except KeyError:
            print(f"Study '{full_study_name}' not found, skipping...")
            continue

        _, _, dataset, method = full_study_name.split("/")

        trials = [
            t
            for t in study.trials
            if t.state
            in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        ][:max_trials]
        if not trials:
            print(
                f"Study '{full_study_name}' has no completed/pruned trials, skipping..."
            )
            continue

        best_trial = max(trials, key=lambda t: t.value or 0)

        # Select the best trial based on the highest score (single objective value)

        accuracy = best_trial.user_attrs["avg_acc"]
        ece = best_trial.user_attrs["avg_ece"]
        score_ = best_trial.values[0]

        print("========================================")
        print(f"Dataset {dataset}, Method: {method}")
        print(f"Avg. Acc. {accuracy * 100:.2f}")
        print(f"ECE       {ece * 100:.2f}")
        print(f"Score:    {score_ * 100:.2f}")
        print(f"N Trials  {len(trials)}")

        git_hash = study.user_attrs.get("git_commit")

        config = {}

        for param in best_trial.params:
            keys = param.split(".")
            sub_config = config
            for key in keys[:-1]:
                sub_config = sub_config.setdefault(key, {})
            sub_config[keys[-1]] = best_trial.params[param]

        config_file = jinja_template.render(
            study_id=study._study_id,
            study_name=full_study_name,
            git_hash=git_hash,
            dataset=dataset,
            method=method,
            accuracy=accuracy,
            ece=ece,
            score=score_,
            n_trials=len(trials),
            hyperparameters=config,
        )
        # print(config_file)

        filename = f"configs/{dataset}/{method}.jsonnet"
        with open(filename, "w") as f:
            f.write(config_file)


if __name__ == "__main__":
    main()
