"""Arm registry.

An *arm* is a single method / treatment: a dataclass holding that method's
hyperparameters, a ``suggest_config`` that samples them from an Optuna trial,
    builder hooks for method-specific construction, and a ``build`` hook that can
    customize a runnable :class:`~bayescl.experiment.Experiment`.

Register variants by subclassing an existing arm and changing field defaults,
then ``@register("name")`` under a new key (see ``tball`` / ``tball_mnd``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import optuna
import torch
from avalanche.training import Cumulative, Naive
from loguru import logger

if TYPE_CHECKING:
    from bayescl.bnn.vcl import VCLConfig
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
    if not ARMS:
        import bayescl.arms  # noqa: F401
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
    def suggest_lr(trial: optuna.Trial) -> float:
        # Floor raised from 1e-4 to 1e-3 after the cifar100/lora pilot: values
        # near the old floor didn't converge within the epoch budget on any
        # task, while every value >= ~2e-3 did (with some margin to spare).
        return trial.suggest_float("lr", 1e-3, 1e-2, log=True)

    @staticmethod
    def suggest_config(trial: optuna.Trial, base: "ArmBase") -> "ArmBase":
        return replace(base, lr=ArmBase.suggest_lr(trial))

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
            from bayescl.plugin.train_mask import TrainTaskMask

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
            train_mb_size=experiment.config.train_mb_size,
            eval_mb_size=experiment.config.eval_mb_size
            or experiment.config.train_mb_size,
            train_epochs=experiment.config.epochs,
            evaluator=experiment.eval_plugin,
            device=experiment.config.device,
            plugins=experiment.plugins,
            eval_every=-1,
            criterion=torch.nn.CrossEntropyLoss(),
        )

    def _build_naive_strategy(self, experiment: "Experiment") -> Any:
        return Naive(**self._strategy_kwargs(experiment))

    def _build_cumulative_strategy(self, experiment: "Experiment") -> Any:
        return Cumulative(**self._strategy_kwargs(experiment))

    def _build_vcl_strategy(self, experiment: "Experiment", config: "VCLConfig") -> Any:
        from bayescl.bnn.vcl import VCLStrategy

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
