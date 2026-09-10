from dataclasses import dataclass, replace

import torch
from bayescl.methods._registry import ArmBase, register
from bayescl.methods.clora import CLoRAAdapterFactory, CLoRAConfig, CLoRAPlugin
from bayescl.peft import RegexFilter, add_adapters
from loguru import logger


@register("clora")
@dataclass
class CLoRA(ArmBase):
    lr: float = 1e-3
    rank: int = 10
    alpha: float = 1.0
    lambda_: float = 1.0

    def _peft(self):
        return CLoRAConfig(rank=self.rank, alpha=self.alpha, lambda_=self.lambda_)

    def _build_peft(self, experiment):
        logger.info("Adding CLoRA adapters and plugin")
        torch.manual_seed(experiment.config.seed + 7808)
        peft = self._peft()
        add_adapters(
            experiment.model,
            RegexFilter(experiment.config.adapter_filter),
            CLoRAAdapterFactory(experiment.num_tasks, peft),
        )
        experiment.plugins.append(CLoRAPlugin(peft, experiment.tb_log.writer))
        experiment.model.get_submodule(experiment.config.head_module).requires_grad_(True)

    def _build_strategy(self, experiment):
        return self._build_naive_strategy(experiment)

    @staticmethod
    def suggest_config(trial, base):
        return replace(
            base,
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            lambda_=trial.suggest_float("lambda_", 0.01, 100.0, log=True),
            alpha=trial.suggest_float("alpha", 0.5, 2.0),
        )
