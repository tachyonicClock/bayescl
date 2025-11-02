from pathlib import Path
from typing import Callable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import csv
from os import environ

import click
import numpy as np
import optuna
from loguru import logger

from bayescl.config import Config, from_configs
from bayescl.experiment import Experiment
from bayescl.util.git import commit_message, commit_short_hash, is_git_status_clean
from bayescl.util.optuna import obj_dot_notation_set, optuna_suggest

OPTUNA_PROJECT_PREFIX = "bayescl"


def get_sampler(sampler: str) -> optuna.samplers.BaseSampler:
    if sampler == "TPE":
        return optuna.samplers.TPESampler()
    elif sampler == "random":
        return optuna.samplers.RandomSampler()
    elif sampler == "QMC":
        return optuna.samplers.QMCSampler(scramble=True)
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


def get_optuna_study_name(config: Config) -> str:
    return f"{OPTUNA_PROJECT_PREFIX}/{config.label.study}/{config.label.scenario}/{config.label.method}"


def run_study(config: Config):
    assert config.hpsearch
    n_trials = config.hpsearch.n_trials
    assert n_trials and n_trials >= 1

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        assert config.hpsearch

        config.label.run = f"trial_{trial.number:04d}"
        config.seed = trial.number
        optuna_suggest(trial, config, config.hpsearch.params)
        return Experiment(config).run(trial)

    study = optuna.create_study(
        directions=config.hpsearch.direction,
        study_name=get_optuna_study_name(config),
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


@click.group()
@click.option(
    "--configs",
    "-c",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config files to load. Can be specified multiple times. The order determines precedence.",
)
@click.option(
    "--args",
    "-a",
    type=str,
    multiple=True,
    help="Override config options using dotlist notation.",
)
@click.option(
    "--epochs-scale",
    help="Coefficient to scale number of epochs by.",
    default=1.0,
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Skip checking if repo is clean.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    configs: List[str],
    args: List[str],
    epochs_scale: float,
    force: bool,
):
    if not force and not is_git_status_clean():
        raise SystemExit("Please ensure everything is committed")

    cfg = from_configs(configs, args)
    if epochs_scale != 1.0:
        epochs = int(epochs_scale * cfg.epochs)
        logger.info(f"Scaling epochs {cfg.epochs} -> {epochs}")
        cfg.epochs = epochs

    ctx.obj = cfg


@cli.command()
@click.option(
    "--validate",
    is_flag=True,
    default=False,
    help="Run in validation mode",
)
@click.option(
    "--from-study",
    is_flag=True,
    default=False,
    help="Use the best configuration found by optuna.",
)
@click.option(
    "--n-trials",
    "-n",
    type=int,
    default=1,
    help="Repeat with different seeds.",
)
@click.pass_obj
def run(
    cfg: Config,
    validate: bool,
    from_study: bool,
    n_trials: int = 1,
):
    cfg.scenario.validation = validate
    git_commit_hash = commit_short_hash()
    git_message = commit_message()

    study = None
    if from_study:
        study = optuna.load_study(
            study_name=get_optuna_study_name(cfg),
            storage=environ["OPTUNA_STORAGE"],
        )
        best_trial = min(study.best_trials, key=lambda t: t.values[1])

        logger.info(f"Using best configuration from study {study.study_name}")
        logger.info(f"Using best trial #{best_trial.number} from study '{study}'")
        logger.info(f"  Values: {best_trial.values}")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")
            obj_dot_notation_set(key, cfg, value)

    accuracy_seen_avgs, ece_seen_avgs = [], []
    for i in range(n_trials):
        cfg.seed = cfg.seed + i
        if n_trials > 1:
            cfg.label.run = f"run_{i:04d}"
        accuracy_seen_avg, ece_seen_avg = Experiment(cfg).run()
        accuracy_seen_avgs.append(accuracy_seen_avg)
        ece_seen_avgs.append(ece_seen_avg)

    if from_study and study is not None:
        log_to_logbook(
            cfg,
            study._study_id,
            accuracy_seen_avgs,
            ece_seen_avgs,
            git_commit_hash,
            git_message,
        )


def log_to_logbook(
    cfg: Config,
    optuna_id: int,
    accuracy_seen_avgs: Sequence[float],
    ece_seen_avgs: Sequence[float],
    git_commit_hash: str,
    git_message: str,
):
    filename = Path(
        f"~/logbooks/{cfg.label.scenario}_{cfg.label.method}.csv"
    ).expanduser()
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "a") as f:
        writer = csv.writer(
            f,
            strict=True,
        )
        # if empty file, write header
        if f.tell() == 0:
            writer.writerow(
                [
                    "scenario",
                    "method",
                    "study",
                    "optuna_id",
                    "git_commit",
                    "n_trials",
                    "accuracy_mean",
                    "accuracy_std",
                    "ece_mean",
                    "ece_std",
                    "git_message",
                ]
            )
        writer.writerow(
            [
                cfg.label.scenario,
                cfg.label.method,
                cfg.label.study,
                f"{optuna_id:06d}",
                git_commit_hash,
                len(accuracy_seen_avgs),
                f"{np.mean(accuracy_seen_avgs) * 100:0.4f}",
                f"{np.std(accuracy_seen_avgs) * 100:0.4f}",
                f"{np.mean(ece_seen_avgs) * 100:0.4f}",
                f"{np.std(ece_seen_avgs) * 100:0.4f}",
                git_message,
            ]
        )


@cli.command()
@click.pass_obj
def hpsearch(cfg: Config):
    cfg.scenario.validation = True
    run_study(cfg)


if __name__ == "__main__":
    cli()
