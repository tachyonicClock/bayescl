from dataclasses import dataclass, replace

import torch
from loguru import logger

from bayescl.bnn.vbnn import VBNNConfig
from bayescl.bnn.vcl import VCLConfig
from bayescl.peft import RegexFilter, add_adapters
from bayescl.treatments._registry import ArmBase, register
from bayescl.treatments.ball import BALLAdapterFactory, BALLConfig


@register("ball")
@dataclass
class BALL(ArmBase):
    lr: float = 1e-3
    r: int = 8
    lora_alpha: int = 1
    dropout: float = 0.0
    bll: bool = False
    # VCL strategy
    beta: float = 1.0
    train_samples: int = 1
    test_samples: int = 5
    # variational prior / init
    prior_mean: float = 0.0
    prior_sd: float = 1.0
    init_sd: float = 1e-3
    init_sd_sd: float = 1e-5

    def _peft(self):
        return BALLConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            dropout=self.dropout,
            bll=self.bll,
            vbnn=VBNNConfig(
                prior_mean=self.prior_mean,
                prior_sd=self.prior_sd,
                init_sd=self.init_sd,
                init_sd_sd=self.init_sd_sd,
            ),
        )

    def _build_peft(self, experiment):
        logger.info("Adding BALL adapters")
        torch.manual_seed(experiment.config.seed + 7808)
        peft = self._peft()
        add_adapters(
            experiment.model,
            RegexFilter(experiment.config.adapter_filter),
            BALLAdapterFactory(peft),
        )
        if peft.bll:
            from bayescl.bnn.vbnn import replace_head

            replace_head(
                experiment.model, experiment.config.head_module, config=peft.vbnn
            )
        experiment.model.get_submodule(experiment.config.head_module).requires_grad_(
            True
        )

    def _build_plugins(self, experiment):
        self._build_common_plugins(experiment, local_ce=False)

    def _build_strategy(self, experiment):
        return self._build_vcl_strategy(
            experiment,
            VCLConfig(
                beta=self.beta,
                train_samples=self.train_samples,
                test_samples=self.test_samples,
            ),
        )

    @staticmethod
    def suggest_config(trial, base):
        return replace(
            base,
            lr=ArmBase.suggest_lr(trial),
            beta=trial.suggest_float("beta", 0.0, 2.0),
        )
