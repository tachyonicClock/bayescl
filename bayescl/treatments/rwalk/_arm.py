from dataclasses import dataclass, replace

import torch
from bayescl.treatments._registry import ArmBase, register
from bayescl.treatments.lora import LoRAAdapterFactory, LoRAConfig
from bayescl.peft import RegexFilter, add_adapters
from avalanche.training.plugins import RWalkPlugin
from loguru import logger


@register("rwalk")
@dataclass
class RWalk(ArmBase):
    lr: float = 1e-3
    r: int = 10
    ewc_lambda: float = 0.1
    ewc_alpha: float = 0.9
    delta_t: int = 10

    def _build_peft(self, experiment):
        logger.info("Adding LoRA adapters")
        torch.manual_seed(experiment.config.seed + 7808)
        add_adapters(
            experiment.model,
            RegexFilter(experiment.config.adapter_filter),
            LoRAAdapterFactory(LoRAConfig(r=self.r)),
        )
        experiment.model.get_submodule(experiment.config.head_module).requires_grad_(True)

    def _build_plugins(self, experiment):
        self._build_common_plugins(
            experiment,
            append_metrics=False,
        )
        logger.info("Add 'RWalk' plugin")
        experiment.plugins.append(
            RWalkPlugin(
                ewc_lambda=self.ewc_lambda,
                ewc_alpha=self.ewc_alpha,
                delta_t=self.delta_t,
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
            ewc_lambda=trial.suggest_float("ewc_lambda", 0.0, 1.0),
            ewc_alpha=trial.suggest_float("ewc_alpha", 0.0, 1.0),
        )
