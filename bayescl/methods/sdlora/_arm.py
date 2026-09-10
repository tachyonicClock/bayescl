from dataclasses import dataclass

import torch
from bayescl.methods._registry import ArmBase, register
from bayescl.methods.sdlora import SDLoRAAdapterFactory, SDLoRAConfig, SDLoRAPlugin
from bayescl.peft import RegexFilter, add_adapters
from loguru import logger


@register("sdlora")
@dataclass
class SDLoRA(ArmBase):
    lr: float = 1e-3
    rank_per_task: int = 1

    def _build_peft(self, experiment):
        logger.info("Adding SD-LoRA adapters")
        torch.manual_seed(experiment.config.seed + 7808)
        peft = SDLoRAConfig(rank_per_task=self.rank_per_task)
        add_adapters(
            experiment.model,
            RegexFilter(experiment.config.adapter_filter),
            SDLoRAAdapterFactory(experiment.num_tasks, peft),
        )
        experiment.plugins.append(SDLoRAPlugin())
        experiment.model.get_submodule(experiment.config.head_module).requires_grad_(True)

    def _build_strategy(self, experiment):
        return self._build_naive_strategy(experiment)
