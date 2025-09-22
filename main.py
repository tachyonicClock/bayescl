from typing import Callable

import matplotlib

matplotlib.use("Agg")

import argparse
from os import environ

import optuna

from bayescl.config import Config, from_configs
from bayescl.experiment import Experiment
from bayescl.util.git import commit_message, commit_short_hash, is_git_status_clean
from bayescl.util.optuna import optuna_suggest


def get_sampler(sampler: str) -> optuna.samplers.BaseSampler:
    if sampler == "TPE":
        return optuna.samplers.TPESampler()
    elif sampler == "random":
        return optuna.samplers.RandomSampler()
    raise ValueError(f"Unknown sampler: {sampler}")


def optimize_with_max_trials(
    study: "optuna.study.Study",
    objective: Callable[[optuna.trial.Trial], tuple[float, ...]],
    n_trials: int,
    states: tuple[optuna.trial.TrialState, ...] = (optuna.trial.TrialState.COMPLETE,),
    callbacks=[],
    # rest optuna options
    **kwargs,
):
    """
    By default the n_trials specifies trials count per worker.
    So if you use multiple processes you will have some issues:
    - you should know exactly how much workers will it be to pick correct value
    - if some of workers will reach it's n_trials faster, you'll get an idle
      worker which could do some work otherwise
    - if you'll restart the process — trial count will start from scratch without
      accounting for earlier finished trials

    Source: https://github.com/optuna/optuna/issues/1883#issuecomment-702688136
    """

    trials = study.get_trials(deepcopy=False, states=states)
    n_complete = len(trials)

    if n_complete >= n_trials:
        return

    callbacks.append(optuna.study.MaxTrialsCallback(n_trials, states=states))

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=callbacks,
        **kwargs,
    )


def run_study(config: Config):
    assert config.hpsearch
    n_trials = config.hpsearch.n_trials
    assert n_trials and n_trials >= 1
    assert is_git_status_clean(), "Please ensure everything is committed"

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        assert config.hpsearch

        config.label.run = f"{trial.number:04d}"
        config.seed = trial.number
        optuna_suggest(trial, config, config.hpsearch.params)
        return Experiment(config).run(trial)

    study = optuna.create_study(
        directions=config.hpsearch.direction,
        study_name=f"bayescl/{config.label.study}/{config.scenario.dataset}/{config.label.method}",
        storage=environ.get("OPTUNA_STORAGE"),
        sampler=get_sampler(config.hpsearch.sampler),
        load_if_exists=True,
    )
    study.set_user_attr("git_commit", commit_short_hash())
    study.set_user_attr("git_message", commit_message())

    optimize_with_max_trials(
        study,
        objective,
        n_trials=n_trials,
        states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.RUNNING),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Accept any number of key value pairs as dotlist arguments
    parser.add_argument("--args", type=str, nargs="*", default=None)
    parser.add_argument(
        "--configs",
        "-c",
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
        seed = config.seed or 0
        for i in range(args.repeat):
            config.seed = i + seed
            Experiment(config).run()  # type: ignore
