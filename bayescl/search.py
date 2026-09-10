"""Helpers for resolving Optuna samplers and pruners."""

from __future__ import annotations

import optuna


def get_sampler(name: str) -> optuna.samplers.BaseSampler:
    factories = {
        "TPESampler": lambda: optuna.samplers.TPESampler(),
        "RandomSampler": lambda: optuna.samplers.RandomSampler(),
        "QMCSampler": lambda: optuna.samplers.QMCSampler(scramble=True),
        "BruteForceSampler": lambda: optuna.samplers.BruteForceSampler(),
    }
    try:
        return factories[name]()
    except KeyError:
        raise ValueError(f"Unknown sampler: {name}") from None


def get_pruner(name: str) -> optuna.pruners.BasePruner:
    factories = {
        "NopPruner": optuna.pruners.NopPruner,
        "MedianPruner": optuna.pruners.MedianPruner,
        "HyperbandPruner": optuna.pruners.HyperbandPruner,
        "SuccessiveHalvingPruner": optuna.pruners.SuccessiveHalvingPruner,
    }
    try:
        return factories[name]()
    except KeyError:
        raise ValueError(f"Unknown pruner: {name}") from None
