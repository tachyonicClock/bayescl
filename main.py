#!/usr/bin/env -S uv run
"""bayescl experiment CLI.

    main.py tune <pilot|full> <dataset> <method>   # search hyperparameters
    main.py test <pilot|full> <dataset> <method>   # evaluate best config over seeds

Artifacts are written to::

    runs/tune/{scale}/{dataset}/{method}/{RUNID}/
        results.jsonl   # one line per trial; consumed by `test`
        meta.json       # provenance + best trial
        trial_XXXX/      # per-trial Experiment output
    runs/test/{scale}/{dataset}/{method}/{RUNID}/
        results.jsonl   # one line per seed
        meta.json
        seed_XX/

RUNID is a %Y-%m-%d_%H-%M-%S timestamp; `test` uses the most recent tune run.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import click
import optuna
from loguru import logger

from bayescl.arms import get_arm
from bayescl.base import NumericError
from bayescl.config import (
    ExperimentConfig,
    dataset_names,
    get_dataset,
    get_scale,
    scale_names,
)
from bayescl.experiment import Experiment
from bayescl.methods._registry import arm_names
from bayescl.runio import append_jsonl, latest_run, read_jsonl, score, write_json
from bayescl.util.git import commit_message, commit_short_hash, is_git_status_clean

_DATASET_PATH = os.environ.get("DATASETS")
_SAMPLER = optuna.samplers.TPESampler()
_PRUNER = optuna.pruners.MedianPruner()


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _targets(f):
    f = click.argument("method", type=click.Choice(arm_names()))(f)
    f = click.argument("dataset", type=click.Choice(dataset_names()))(f)
    f = click.argument("scale", type=click.Choice(scale_names()))(f)
    f = click.option(
        "--runs",
        type=click.Path(file_okay=False),
        default="./runs",
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
    return f


def _meta(stage, scale, dataset, method, runid, sc, ds, **extra) -> dict:
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
            "epochs": sc.epochs(ds),
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
def tune(scale, dataset, method, runs, dataset_path, device, sqlite):
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
        ),
    )

    study = optuna.create_study(
        direction="maximize",
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
                ds, sc,
                seed=trial.number,
                validation=True,
                run_dir=run_dir / f"trial_{trial.number:04d}",
                dataset_root=Path(dataset_path),
                device=device,
            ),
            arm,
        )
        row = {
            "trial": trial.number,
            "seed": trial.number,
            "arm": asdict(arm),
            "ts": _timestamp(),
        }
        try:
            acc, ece = exp.run(trial)
        except optuna.TrialPruned:
            append_jsonl(
                results,
                {
                    **row,
                    "params": trial.params,
                    "state": "pruned",
                    "acc": None,
                    "ece": None,
                    "score": None,
                },
            )
            raise
        except NumericError as e:
            append_jsonl(
                results,
                {
                    **row,
                    "params": trial.params,
                    "state": "failed",
                    "error": str(e),
                    "acc": None,
                    "ece": None,
                    "score": None,
                },
            )
            raise
        s = score(acc, ece)
        trial.set_user_attr("acc", acc)
        trial.set_user_attr("ece", ece)
        append_jsonl(
            results,
            {
                **row,
                "params": trial.params,
                "state": "complete",
                "acc": acc,
                "ece": ece,
                "score": s,
            },
        )
        return s

    study.optimize(objective, n_trials=sc.n_trials, catch=(NumericError,))

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
            finished=_timestamp(),
            best={
                "trial": best.number,
                "params": best.params,
                "acc": best.user_attrs.get("acc"),
                "ece": best.user_attrs.get("ece"),
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
def test(scale, dataset, method, runs, dataset_path, device, from_tune):
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
    best = max(rows, key=lambda r: r["score"])
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
            source_tune=str(tune_dir),
            best_trial=best["trial"],
            best_params=best["params"],
            arm=best["arm"],
        ),
    )

    for seed in range(sc.n_seeds):
        exp = Experiment(
            ExperimentConfig.from_spec(
                ds, sc,
                seed=seed,
                validation=False,
                run_dir=run_dir / f"seed_{seed:02d}",
                dataset_root=Path(dataset_path),
                device=device,
            ),
            arm,
        )
        acc, ece = exp.run(None)
        append_jsonl(
            results,
            {
                "seed": seed,
                "acc": acc,
                "ece": ece,
                "score": score(acc, ece),
                "ts": _timestamp(),
            },
        )
        logger.info(f"seed {seed}: acc={acc:.4f} ece={ece:.4f}")


if __name__ == "__main__":
    main()
