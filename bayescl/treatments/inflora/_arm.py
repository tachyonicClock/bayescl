from dataclasses import dataclass, replace

import torch
from bayescl.treatments._registry import ArmBase, register
from bayescl.treatments.inflora import InfLoRAAdapterFactory, InfLoRAConfig, InfLoRAPlugin
from bayescl.peft import RegexFilter, add_adapters
from loguru import logger


@register("inflora")
@dataclass
class InfLoRA(ArmBase):
    lr: float = 1e-3
    rank: int = 8
    threshold_start: float = 0.90
    threshold_end: float = 0.98
    max_activation_batches: int | None = 16
    activation_batch_size: int = 32

    def _peft(self):
        return InfLoRAConfig(
            rank=self.rank,
            threshold_start=self.threshold_start,
            threshold_end=self.threshold_end,
            max_activation_batches=self.max_activation_batches,
            activation_batch_size=self.activation_batch_size,
        )

    def _build_peft(self, experiment):
        logger.info("Adding InfLoRA adapters and plugin")
        torch.manual_seed(experiment.config.seed + 7808)
        peft = self._peft()
        add_adapters(
            experiment.model,
            RegexFilter(experiment.config.adapter_filter),
            InfLoRAAdapterFactory(peft),
        )
        experiment.plugins.append(
            InfLoRAPlugin(
                peft,
                total_tasks=experiment.num_tasks,
                optimizer_factory=self.configure_optimizers,
                max_activation_batches=peft.max_activation_batches,
            )
        )
        experiment.model.get_submodule(experiment.config.head_module).requires_grad_(True)

    def _build_strategy(self, experiment):
        return self._build_naive_strategy(experiment)

    @staticmethod
    def suggest_config(trial, base):
        threshold_start = trial.suggest_float("threshold_start", 0.80, 0.99)
        threshold_end = trial.suggest_float("threshold_end", threshold_start, 0.999)
        return replace(
            base,
            lr=ArmBase.suggest_lr(trial),
            threshold_start=threshold_start,
            threshold_end=threshold_end,
        )
