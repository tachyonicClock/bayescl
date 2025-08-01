import matplotlib

matplotlib.use("Agg")

import argparse
from os import environ

import optuna
from claiutil.git import commit_short_hash, is_git_status_clean
from claiutil.optuna import optuna_suggest

from bayescl.config import Config, from_configs
from bayescl.experiment import Experiment


def run_study(config: Config):
    assert config.hpsearch
    assert is_git_status_clean(), "Please ensure everything is committed"

    def objective(trial: optuna.Trial) -> float:
        assert config.hpsearch
        optuna_suggest(trial, config, config.hpsearch.params)
        return Experiment(config).run()

    study = optuna.create_study(
        direction=config.hpsearch.direction,
        study_name=f"bayescl/{config.scenario.dataset}/{config.label}/{commit_short_hash}",
        storage=environ.get("OPTUNA_STORAGE"),
        load_if_exists=True,
    )
    config.study_name = "hpsearch"
    study.optimize(objective, n_trials=config.hpsearch.n_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Accept any number of key value pairs as dotlist arguments
    parser.add_argument("--args", type=str, nargs="*", default=None)
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=[],
        help="Path to config files to load. Can be specified multiple times.",
    )
    parser.add_argument(
        "--hpsearch",
        action="store_true",
        default=False,
        help="Enable hyperparameter search.",
    )
    args = parser.parse_args()
    config = from_configs(args.configs, args.args)

    if args.hpsearch is not None:
        run_study(config)
    else:
        for _ in range(config.repeat):
            Experiment(config).run()  # type: ignore
