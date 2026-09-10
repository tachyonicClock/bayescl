from dataclasses import dataclass, replace

import torch
from avalanche.training.plugins import EWCPlugin
from loguru import logger

from bayescl.peft import RegexFilter, add_adapters
from bayescl.treatments._registry import ArmBase, register
from bayescl.treatments.lora import LoRAAdapterFactory, LoRAConfig


@register("ewc")
@dataclass
class EWC(ArmBase):
    lr: float = 1e-3
    r: int = 10
    ewc_lambda: float = 1.0
    decay_factor: float = 0.9

    def _build_peft(self, experiment):
        logger.info("Adding LoRA adapters")
        torch.manual_seed(experiment.config.seed + 7808)
        add_adapters(
            experiment.model,
            RegexFilter(experiment.config.adapter_filter),
            LoRAAdapterFactory(LoRAConfig(r=self.r)),
        )
        experiment.model.get_submodule(experiment.config.head_module).requires_grad_(
            True
        )

    def _build_plugins(self, experiment):
        self._build_common_plugins(
            experiment,
            append_metrics=False,
        )
        logger.info("Add 'EWCPlugin' plugin")
        experiment.plugins.append(
            EWCPlugin(
                ewc_lambda=self.ewc_lambda,
                decay_factor=self.decay_factor,
                mode="online",
            )
        )
        experiment.plugins.append(experiment.metrics_plugin)

    def _build_strategy(self, experiment):
        return self._build_naive_strategy(experiment)

    @staticmethod
    def suggest_config(trial, base):
        return replace(
            base,
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            ewc_lambda=trial.suggest_float("ewc_lambda", 0.0, 10.0),
            decay_factor=trial.suggest_float("decay_factor", 0.8, 1.0),
        )
