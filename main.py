from typing import Callable, List

import matplotlib

from bayescl.base import NumericError

matplotlib.use("Agg")

from os import environ

import click
import optuna

from bayescl.config import Config, ZeusMonitorConfig, from_config
from bayescl.experiment import Experiment
from bayescl.util.git import commit_message, commit_short_hash, is_git_status_clean
from bayescl.util.optuna import optuna_suggest

OPTUNA_PROJECT_PREFIX = "bayescl"


def get_sampler(sampler: str) -> optuna.samplers.BaseSampler:
    if sampler == "TPE":
        return optuna.samplers.TPESampler()
    elif sampler == "random":
        return optuna.samplers.RandomSampler()
    elif sampler == "BruteForceSampler":
        return optuna.samplers.BruteForceSampler()
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
        catch=[NumericError],
        **kwargs,
    )


def get_optuna_study_name(config: Config) -> str:
    return f"{OPTUNA_PROJECT_PREFIX}/{config.label.study}/{config.label.scenario}/{config.label.method}"


def run_study(config: Config):
    assert config.hpsearch
    n_trials = config.hpsearch.n_trials
    assert n_trials and n_trials >= 1

    def objective(trial: optuna.Trial) -> tuple[float, ...] | float:
        assert config.hpsearch

        config.label.run = f"trial_{trial.number:04d}"
        config.seed = trial.number
        optuna_suggest(trial, config, config.hpsearch.params)
        avg_acc, avg_ece = Experiment(config).run(trial)
        trial.set_user_attr("avg_acc", avg_acc)
        trial.set_user_attr("avg_ece", avg_ece)
        if len(config.hpsearch.direction) == 1:
            return 0.5 * (avg_acc + (1 - avg_ece))
        return avg_acc, avg_ece

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
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a jsonnet config file.",
)
@click.option(
    "--args",
    "-a",
    type=str,
    multiple=True,
    help="Override config options using dotlist notation.",
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
    config: str,
    args: List[str],
    force: bool,
):
    if not force and not is_git_status_clean():
        raise SystemExit("Please ensure everything is committed")

    cfg = from_config(config, args)
    ctx.obj = cfg


@cli.command()
@click.option(
    "--validate",
    is_flag=True,
    default=False,
    help="Run in validation mode",
)
@click.option(
    "--zeus",
    is_flag=True,
    default=False,
    help="Enable Zeus GPU energy monitoring.",
)
@click.argument("name", type=str, default="manual")
@click.argument("seed", type=int, default=0)
@click.pass_obj
def run(
    cfg: Config,
    name: str,
    seed: int,
    validate: bool,
    zeus: bool,
):
    cfg.scenario.validation = validate
    cfg.seed = seed
    cfg.label.study = name
    cfg.label.run = f"{seed:02d}"
    if zeus and cfg.zeus_monitor is None:
        cfg.zeus_monitor = ZeusMonitorConfig()
    Experiment(cfg).run()


@cli.command()
@click.pass_obj
@click.argument("name", type=str)
def hpsearch(cfg: Config, name: str):
    cfg.label.study = name
    run_study(cfg)


@cli.command()
@click.pass_obj
def count_parameters(cfg: Config):
    experiment = Experiment(cfg)
    experiment.count_parameters()


if __name__ == "__main__":
    cli()
