from dataclasses import dataclass, replace

import torch
from loguru import logger

from bayescl.bnn.vcl import VCLConfig
from bayescl.peft import RegexFilter, add_adapters
from bayescl.treatments._registry import ArmBase, register
from bayescl.treatments.tball import TBALLAdapterFactory, TBALLConfig


@register("tball")
@dataclass
class TBALL(ArmBase):
    lr: float = 1e-3
    rank: int = 10
    alpha: float = 1.0
    prior_mean: float = 0.0
    prior_weight_sd: float = 1.0
    init_sd: float = 0.5
    nonlinearity_scale: float = 1.0
    bnn: str = "FFG"
    bias: bool = False
    # VCL strategy
    beta: float = 1.0
    train_samples: int = 1
    test_samples: int = 5

    def _build_peft(self, experiment):
        logger.info("Adding TBALL adapters")
        torch.manual_seed(experiment.config.seed + 7808)
        add_adapters(
            experiment.model,
            RegexFilter(experiment.config.adapter_filter),
            TBALLAdapterFactory(self._peft()),
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
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            beta=trial.suggest_float("beta", 0.0, 2.0),
        )

    def _peft(self) -> TBALLConfig:
        return TBALLConfig(
            rank=self.rank,
            alpha=self.alpha,
            prior_mean=self.prior_mean,
            prior_weight_sd=self.prior_weight_sd,
            init_sd=self.init_sd,
            nonlinearity_scale=self.nonlinearity_scale,
            bnn=self.bnn,
            bias=self.bias,
        )


@register("tball_mnd")
@dataclass
class TBALLMND(TBALL):
    # Registry variant: identical to TBALL except the Bayesian core is a matrix
    # normal distribution. ``suggest_config`` / ``build`` are inherited unchanged.
    bnn: str = "MND"
