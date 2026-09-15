#!/usr/bin/env -S uv run
"""bayescl experiment CLI.

    main.py tune <pilot|full> <dataset> <method>   # search hyperparameters
    main.py test <pilot|full> <dataset> <method>   # evaluate best config over seeds

Artifacts are written to::

    $LOGDIR/bayescl/tune/{scale}/{dataset}/{method}/{RUNID}/
        results.jsonl   # one line per trial; consumed by `test`
        meta.json       # provenance + best trial
        trial_XXXX/      # per-trial Experiment output
    $LOGDIR/bayescl/test/{scale}/{dataset}/{method}/{RUNID}/
        results.jsonl   # one line per seed
        meta.json
        seed_XX/

RUNID is a %Y-%m-%d_%H-%M-%S timestamp; `test` uses the most recent tune run.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import click
import optuna
from loguru import logger

from bayescl.arms import get_arm
from bayescl.base import ConvergenceError, NumericError
from bayescl.config import (
    ExperimentConfig,
    dataset_names,
    get_dataset,
    get_scale,
    scale_names,
)
from bayescl.experiment import Experiment
from bayescl.git import commit_message, commit_short_hash, is_git_status_clean
from bayescl.metrics.results import Result
from bayescl.runio import append_jsonl, latest_run, read_jsonl, write_json
from bayescl.treatments._registry import arm_names


class _InterceptHandler(logging.Handler):
    """Route stdlib ``logging`` records (Optuna's, e.g.) through loguru so
    they share this process's log format instead of their own."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk past logging's own frames so loguru reports the record's real
        # call site (e.g. optuna's) rather than this handler's ``emit``.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# Optuna installs its own handler on its logger with propagation disabled;
# swap that for propagation to the (now intercepted) root logger instead.
optuna.logging.disable_default_handler()
optuna.logging.enable_propagation()
logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO, force=True)

_DATASET_PATH = os.environ.get("DATASETS")
_RUNS_PATH = Path(os.environ.get("LOGDIR", "logs")) / "bayescl"
# multivariate=True models interactions between hyperparameters (e.g. lr vs.
# LoRA alpha) instead of treating them independently, converging faster.
_SAMPLER = optuna.samplers.TPESampler(multivariate=True)
_PRUNER = optuna.pruners.MedianPruner()
# All trials in a tune run share this seed (model init + task order) so TPE
# compares hyperparameters against a fixed target instead of also having to
# average out seed variance. Multi-seed evaluation happens in `test`.
_TUNE_SEED = 0


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def record_result(
    path: Path,
    result: Result | None,
    *,
    row: dict,
    trial: optuna.Trial | None = None,
    state: str = "complete",
    error: str | None = None,
) -> float | None:
    """Append metrics and experiment-level OOD/shift aggregates."""
    brier = result.brier_seen_avg if result is not None else None
    if trial is not None:
        trial.set_user_attr("brier", brier)
    result_values = asdict(result) if result is not None else {}
    # filter to only float values
    result_values = {k: v for k, v in result_values.items() if isinstance(v, float)}
    append_jsonl(
        path,
        {
            **row,
            **result_values,
            "params": trial.params if trial is not None else None,
            "state": state,
            "error": error,
            "brier": brier,
            "score": brier,
        },
    )
    return brier


def _targets(f):
    f = click.argument("method", type=click.Choice(arm_names()))(f)
    f = click.argument("dataset", type=click.Choice(dataset_names()))(f)
    f = click.argument("scale", type=click.Choice(scale_names()))(f)
    f = click.option(
        "--runs",
        type=click.Path(file_okay=False),
        default=str(_RUNS_PATH),
        show_default=True,
        help="Directory to store run outputs.",
    )(f)
    f = click.option(
        "--dataset-path",
        type=click.Path(file_okay=False),
        default=_DATASET_PATH,
        help="Path to the datasets directory (defaults to $DATASETS).",
    )(f)
    f = click.option(
        "--device",
        default="cuda",
        show_default=True,
        help="Torch device for training.",
    )(f)
    f = click.option(
        "--epochs",
        type=click.IntRange(min=1),
        default=None,
        help="Override the scale's number of epochs.",
    )(f)
    return f


def _meta(stage, scale, dataset, method, runid, sc, ds, *, epochs=None, **extra) -> dict:
    return {
        "stage": stage,
        "scale": scale,
        "dataset": dataset,
        "method": method,
        "runid": runid,
        "argv": sys.argv,
        "git": {
            "hash": commit_short_hash(),
            "message": commit_message(),
            "dirty": not is_git_status_clean(),
        },
        "scale_knobs": {
            "n_trials": sc.n_trials,
            "n_seeds": sc.n_seeds,
            "epochs": sc.epochs(ds) if epochs is None else epochs,
        },
        "created": _timestamp(),
        **extra,
    }


@click.group()
def main() -> None:
    """Run hyperparameter search (`tune`) and evaluation (`test`)."""


@main.command()
@_targets
@click.option(
    "--sqlite",
    is_flag=True,
    default=False,
    help="Also write optuna.db under the run dir for optuna-dashboard.",
)
def tune(scale, dataset, method, runs, dataset_path, device, epochs, sqlite):
    """Search hyperparameters for METHOD on DATASET at the given SCALE."""
    if dataset_path is None:
        raise click.ClickException("Set $DATASETS or pass --dataset-path.")

    sc, ds, arm_cls = get_scale(scale), get_dataset(dataset), get_arm(method)
    runid = _timestamp()
    run_dir = Path(runs) / "tune" / scale / dataset / method / runid
    run_dir.mkdir(parents=True, exist_ok=False)
    results = run_dir / "results.jsonl"
    meta_path = run_dir / "meta.json"

    base = arm_cls()
    write_json(
        meta_path,
        _meta(
            "tune",
            scale,
            dataset,
            method,
            runid,
            sc,
            ds,
            epochs=epochs,
        ),
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=_SAMPLER,
        pruner=_PRUNER,
        study_name=f"bayescl/{scale}/{dataset}/{method}",
        storage=f"sqlite:///{run_dir / 'optuna.db'}" if sqlite else None,
        load_if_exists=bool(sqlite),
    )

    def objective(trial: optuna.Trial) -> float:
        arm = type(base).suggest_config(trial, base)
        exp = Experiment(
            ExperimentConfig.from_spec(
                ds,
                sc,
                epochs=epochs,
                seed=_TUNE_SEED,
                validation=True,
                run_dir=run_dir / f"trial_{trial.number:04d}",
                dataset_root=Path(dataset_path),
                device=device,
            ),
            arm,
        )
        row = {
            "trial": trial.number,
            "seed": _TUNE_SEED,
            "arm": asdict(arm),
            "ts": _timestamp(),
        }
        try:
            result = exp.run(trial)
        except optuna.TrialPruned:
            logger.warning(f"Trial {trial.number} pruned")
            record_result(results, None, row=row, trial=trial, state="pruned")
            raise
        except (NumericError, ConvergenceError) as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            record_result(
                results, None, row=row, trial=trial, state="failed", error=str(e)
            )
            raise
        return record_result(results, result, row=row, trial=trial)

    study.optimize(
        objective, n_trials=sc.n_trials, catch=(NumericError, ConvergenceError)
    )

    best = study.best_trial
    write_json(
        meta_path,
        _meta(
            "tune",
            scale,
            dataset,
            method,
            runid,
            sc,
            ds,
            epochs=epochs,
            finished=_timestamp(),
            best={
                "trial": best.number,
                "params": best.params,
                "brier": best.user_attrs.get("brier"),
                "score": best.value,
            },
        ),
    )
    logger.info(
        f"Best trial {best.number}: score={best.value:.4f} params={best.params}"
    )


@main.command()
@_targets
@click.option(
    "--from-tune",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Tune run directory to read hyperparameters from (default: latest).",
)
def test(scale, dataset, method, runs, dataset_path, device, epochs, from_tune):
    """Evaluate METHOD's best tuned config on DATASET over `n_seeds` seeds."""
    if dataset_path is None:
        raise click.ClickException("Set $DATASETS or pass --dataset-path.")

    sc, ds, arm_cls = get_scale(scale), get_dataset(dataset), get_arm(method)
    tune_dir = (
        Path(from_tune)
        if from_tune
        else latest_run(Path(runs) / "tune" / scale / dataset / method)
    )
    rows = [
        r for r in read_jsonl(tune_dir / "results.jsonl") if r["state"] == "complete"
    ]
    if not rows:
        raise SystemExit(f"No complete trials in {tune_dir / 'results.jsonl'}")
    best = min(rows, key=lambda r: r["score"])
    arm = arm_cls(**best["arm"])
    logger.info(
        f"Loaded best config from {tune_dir} (trial {best['trial']}): {best['arm']}"
    )

    runid = _timestamp()
    run_dir = Path(runs) / "test" / scale / dataset / method / runid
    run_dir.mkdir(parents=True, exist_ok=False)
    results = run_dir / "results.jsonl"
    write_json(
        run_dir / "meta.json",
        _meta(
            "test",
            scale,
            dataset,
            method,
            runid,
            sc,
            ds,
            epochs=epochs,
            source_tune=str(tune_dir),
            best_trial=best["trial"],
            best_params=best["params"],
            arm=best["arm"],
        ),
    )

    for seed in range(sc.n_seeds):
        exp = Experiment(
            ExperimentConfig.from_spec(
                ds,
                sc,
                epochs=epochs,
                seed=seed,
                validation=False,
                run_dir=run_dir / f"seed_{seed:02d}",
                dataset_root=Path(dataset_path),
                device=device,
            ),
            arm,
        )
        result = exp.run(None)
        brier = record_result(
            results,
            result,
            row={"seed": seed, "ts": _timestamp()},
        )
        logger.info(f"seed {seed}: brier={brier:.4f}")


if __name__ == "__main__":
    main()
