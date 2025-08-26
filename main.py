import matplotlib

matplotlib.use("Agg")

import argparse
from os import environ

import numpy as np
import optuna
import torch
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
        study_name=f"bayescl/{config.scenario.dataset}/{config.label}/{commit_short_hash()}",
        storage=environ.get("OPTUNA_STORAGE"),
        sampler=optuna.samplers.RandomSampler(),
        load_if_exists=True,
    )
    config.study_name = "hpsearch"
    study.optimize(objective, n_trials=config.hpsearch.n_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Accept any number of key value pairs as dotlist arguments
    parser.add_argument("--args", type=str, nargs="*", default=None)
    parser.add_argument(
        "--configs", "-c",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the experiment with different seeds.",
    )
    args = parser.parse_args()
    config = from_configs(args.configs, args.args)

    if args.hpsearch:
        run_study(config)
    else:
        for i in range(args.repeat):
            torch.manual_seed(args.seed + i)
            np.random.seed(args.seed + i)
            Experiment(config).run()  # type: ignore
