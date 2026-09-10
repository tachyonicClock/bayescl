"""Arm registry.

An *arm* is a single method / treatment: a dataclass holding that method's
hyperparameters, a ``suggest_config`` that samples them from an Optuna trial,
    builder hooks for method-specific construction, and a ``build`` hook that can
    customize a runnable :class:`~bayescl.experiment.Experiment`.

Register variants by subclassing an existing arm and changing field defaults,
then ``@register("name")`` under a new key (see ``tball`` / ``tball-mnd``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import torch
import optuna
from avalanche.training import Naive
from loguru import logger

if TYPE_CHECKING:
    from bayescl.experiment import Experiment
    from bayescl.vcl import VCLConfig

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
    hyperparameters and optionally override :meth:`suggest_config` or
    :meth:`build`.
    """

    name: ClassVar[str] = "?"
    lr: float = 1e-3

    # ---- hyperparameter search ----

    @staticmethod
    def suggest_config(trial: optuna.Trial, base: "ArmBase") -> "ArmBase":
        return replace(base, lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True))

    # ---- experiment assembly ----

    def configure_optimizers(self, parameters) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            filter(lambda parameter: parameter.requires_grad, parameters), lr=self.lr
        )

    def _build_peft(self, experiment: "Experiment") -> None:
        raise NotImplementedError

    def _build_plugins(self, experiment: "Experiment") -> None:
        self._build_common_plugins(experiment)

    def _build_strategy(self, experiment: "Experiment") -> Any:
        raise NotImplementedError

    def _build_common_plugins(
        self,
        experiment: "Experiment",
        *,
        local_ce: bool = True,
        append_metrics: bool = True,
    ) -> None:
        if local_ce:
            from bayescl.train_mask import TrainTaskMask

            logger.info("Add 'TrainTaskMask' plugin")
            experiment.plugins.append(
                TrainTaskMask(experiment.mask, self.configure_optimizers)
            )
        if append_metrics:
            experiment.plugins.append(experiment.metrics_plugin)

    def _strategy_kwargs(self, experiment: "Experiment") -> dict[str, Any]:
        return dict(
            model=experiment.model,
            optimizer=self.configure_optimizers(experiment.model.parameters()),
            train_mb_size=experiment.train_mb_size,
            eval_mb_size=experiment.eval_mb_size or experiment.train_mb_size,
            train_epochs=experiment.epochs,
            evaluator=experiment.eval_plugin,
            device=experiment.device,
            plugins=experiment.plugins,
            eval_every=experiment.eval_every,
            criterion=torch.nn.CrossEntropyLoss(),
        )

    def _build_naive_strategy(self, experiment: "Experiment") -> Any:
        return Naive(**self._strategy_kwargs(experiment))

    def _build_vcl_strategy(
        self, experiment: "Experiment", config: "VCLConfig"
    ) -> Any:
        from bayescl.vcl import VCLStrategy

        logger.info("Using Variational Continual Learning (VCL) strategy")
        return VCLStrategy(
            config=config,
            mask=experiment.mask,
            writer=experiment.tb_log.writer,
            optimizer_fn=self.configure_optimizers,
            **self._strategy_kwargs(experiment),
        )

    def build(self, experiment: "Experiment") -> "Experiment":
        return experiment
