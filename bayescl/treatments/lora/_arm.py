from dataclasses import dataclass

import torch
from bayescl.treatments._registry import ArmBase, register
from bayescl.treatments.lora import LoRAAdapterFactory, LoRAConfig
from bayescl.peft import RegexFilter, add_adapters
from loguru import logger


@register("lora")
@dataclass
class LoRA(ArmBase):
    lr: float = 1e-3
    r: int = 8
    lora_alpha: int = 1
    lora_dropout: float = 0.0

    def _build_peft(self, experiment):
        logger.info("Adding LoRA adapters")
        torch.manual_seed(experiment.config.seed + 7808)
        add_adapters(
            experiment.model,
            RegexFilter(experiment.config.adapter_filter),
            LoRAAdapterFactory(
                LoRAConfig(
                    r=self.r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                )
            ),
        )
        experiment.model.get_submodule(experiment.config.head_module).requires_grad_(
            True
        )

    def _build_strategy(self, experiment):
        return self._build_naive_strategy(experiment)
