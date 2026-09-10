from dataclasses import dataclass

import torch
from bayescl.methods._registry import ArmBase, register
from bayescl.methods.inflora import InfLoRAAdapterFactory, InfLoRAConfig, InfLoRAPlugin
from bayescl.peft import RegexFilter, add_adapters
from loguru import logger


@register("inflora")
@dataclass
class InfLoRA(ArmBase):
    lr: float = 1e-3
    rank: int = 10
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
        torch.manual_seed(experiment.spec.seed + 7808)
        peft = self._peft()
        add_adapters(
            experiment.model,
            RegexFilter(experiment.spec.adapter_filter),
            InfLoRAAdapterFactory(peft),
        )
        experiment.plugins.append(
            InfLoRAPlugin(
                peft,
                total_tasks=experiment.num_tasks,
                optimizer_factory=experiment._new_optimizer,
                max_activation_batches=peft.max_activation_batches,
            )
        )
        experiment.model.get_submodule(experiment.spec.head_module).requires_grad_(True)

    def _build_strategy(self, experiment):
        return self._build_naive_strategy(experiment)
