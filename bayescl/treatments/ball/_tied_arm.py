from dataclasses import dataclass, replace

import torch
from loguru import logger

from bayescl.bnn.vcl import VCLConfig
from bayescl.peft import RegexFilter, add_adapters
from bayescl.treatments._registry import ArmBase, register
from bayescl.treatments.ball._tied_config import TiedBALLConfig
from bayescl.treatments.ball._tied_module import TiedBALLAdapterFactory


@register("ball_tied")
@dataclass
class TiedBALL(ArmBase):
    """BALL with the two LoRA factors sharing one rank-space covariance.

    Pairs with the ``ball`` arm as an ablation: same adapter placement, rank,
    scaling and VCL strategy, differing only in the shape of the variational
    posterior.
    """

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

    def _peft(self) -> TiedBALLConfig:
        return TiedBALLConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            dropout=self.dropout,
            bll=self.bll,
            prior_mean=self.prior_mean,
            prior_sd=self.prior_sd,
            init_sd=self.init_sd,
        )

    def _build_peft(self, experiment):
        logger.info("Adding tied BALL adapters")
        torch.manual_seed(experiment.config.seed + 7808)
        peft = self._peft()
        add_adapters(
            experiment.model,
            RegexFilter(experiment.config.adapter_filter),
            TiedBALLAdapterFactory(peft),
        )
        if peft.bll:
            from bayescl.bnn.vbnn import VBNNConfig, replace_head

            replace_head(
                experiment.model,
                experiment.config.head_module,
                config=VBNNConfig(
                    prior_mean=peft.prior_mean,
                    prior_sd=peft.prior_sd,
                    init_sd=peft.init_sd,
                ),
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
