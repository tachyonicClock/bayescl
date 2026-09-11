from dataclasses import dataclass

import torch
from loguru import logger

from bayescl.peft import RegexFilter, add_adapters
from bayescl.treatments._registry import ArmBase, register
from bayescl.treatments.lora import LoRAAdapterFactory, LoRAConfig


@register("lora_joint")
@dataclass
class LoRAJoint(ArmBase):
    lr: float = 1e-3
    r: int = 10
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
        experiment.model.get_submodule(experiment.config.head_module).requires_grad_(True)

    def _build_plugins(self, experiment):
        # Every minibatch already contains every class seen so far (the
        # dataset is cumulative), so the per-task output mask other arms use
        # would incorrectly zero out earlier tasks' classes.
        self._build_common_plugins(experiment, local_ce=False)

    def _build_strategy(self, experiment):
        return self._build_cumulative_strategy(experiment)
