import copy
from dataclasses import dataclass

import torch
from loguru import logger

from bayescl.peft import RegexFilter, add_adapters
from bayescl.treatments._registry import ArmBase, register
from bayescl.treatments.lora import LoRAAdapterFactory, LoRAConfig

from ._ensemble import EnsembleStrategy
from ._module import EnsembleModule


@register("lora_ensemble")
@dataclass
class LoRAEnsemble(ArmBase):
    lr: float = 1e-3
    r: int = 8
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    num_members: int = 5

    def _build_peft(self, experiment):
        logger.info(f"Adding {self.num_members} independent LoRA ensemble members")
        members = []
        for i in range(self.num_members):
            member = copy.deepcopy(experiment.model)
            torch.manual_seed(experiment.config.seed + 7808 + i)
            add_adapters(
                member,
                RegexFilter(experiment.config.adapter_filter),
                LoRAAdapterFactory(
                    LoRAConfig(
                        r=self.r,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=self.lora_dropout,
                    )
                ),
            )
            member.get_submodule(experiment.config.head_module).requires_grad_(True)
            members.append(member)
        experiment.model = EnsembleModule(members)

    def _build_plugins(self, experiment):
        # Masking and optimization are handled per-member inside
        # `EnsembleStrategy`, so skip the shared `TrainTaskMask` plugin.
        self._build_common_plugins(experiment, local_ce=False)

    def _build_strategy(self, experiment):
        return EnsembleStrategy(
            mask=experiment.mask,
            optimizer_fn=self.configure_optimizers,
            **self._strategy_kwargs(experiment),
        )
