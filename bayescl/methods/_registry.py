"""Arm registry.

An *arm* is a single method / treatment: a dataclass holding that method's
hyperparameters, an Optuna search space, and a ``build`` that assembles an
:class:`~bayescl.spec.ExperimentSpec` and returns a runnable
:class:`~bayescl.experiment.Experiment`.

Register variants by subclassing an existing arm and changing field defaults,
then ``@register("name")`` under a new key (see ``tball`` / ``tball-mnd``).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar

import optuna

from bayescl.datasets_spec import Dataset
from bayescl.scale import Scale
from bayescl.search import get_pruner, get_sampler

if TYPE_CHECKING:
    from bayescl.experiment import Experiment

ARMS: dict[str, type["ArmBase"]] = {}


def register(name: str) -> Callable[[type["ArmBase"]], type["ArmBase"]]:
    def deco(cls: type["ArmBase"]) -> type["ArmBase"]:
        if name in ARMS:
            raise ValueError(f"Arm {name!r} is already registered")
        cls.name = name
        ARMS[name] = cls
        return cls

    return deco


def get_arm(name: str) -> type["ArmBase"]:
    return ARMS[name]


def arm_names() -> list[str]:
    return sorted(ARMS)


class ArmBase:
    """Shared behaviour for every arm.

    Subclasses are ``@dataclass`` instances that redeclare ``lr`` plus their own
    hyperparameters, optionally override :meth:`suggest_config`, and implement
    :meth:`build`.
    """

    name: ClassVar[str] = "?"
    lr: float = 1e-3

    # ---- hyperparameter search ----

    @staticmethod
    def suggest_config(trial: optuna.Trial, base: "ArmBase") -> "ArmBase":
        return replace(base, lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True))

    # ---- spec assembly ----

    def base_spec(
        self,
        *,
        dataset: Dataset,
        scale: Scale,
        seed: int,
        validation: bool,
        run_dir: Path,
        dataset_root: Path,
        use_local_ce: bool = True,
    ) -> dict:
        bb = dataset.backbone
        return dict(
            dataset=dataset.scenario,
            n_tasks=dataset.n_tasks,
            shuffle=dataset.shuffle,
            dataset_root=Path(dataset_root),
            standardize=dataset.standardize,
            validation=validation,
            backbone_name=bb.name,
            freeze_backbone=bb.freeze_backbone,
            adapter_filter=bb.adapter_filter,
            head_module=bb.head_module,
            lr=self.lr,
            epochs=scale.epochs(dataset),
            train_mb_size=dataset.train_mb_size,
            eval_mb_size=dataset.eval_mb_size,
            num_workers=dataset.num_workers,
            seed=seed,
            run_dir=run_dir,
            use_local_ce=use_local_ce,
        )

    def build(
        self,
        *,
        dataset: Dataset,
        scale: Scale,
        seed: int,
        validation: bool,
        run_dir: Path,
        dataset_root: Path,
    ) -> "Experiment":
        raise NotImplementedError
