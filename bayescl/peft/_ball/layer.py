import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from bayescl.batch_ensemble import BayesianBatchEnsembleLinear
from bayescl.peft._ball.config import BALLConfig
from bayescl.peft._base import AdapterBase


class BALLLayer(AdapterBase):
    adapter_parameter_names = (
        "ball_A.weight.mu",
        "ball_A.weight.rho",
        "ball_B.weight.mu",
        "ball_B.weight.rho",
    )
    """Tunable parameters in BALL adapters."""


class BALLLinear(nn.Linear, BALLLayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: BALLConfig,
        **kwargs,
    ):
        self.config = config
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        BALLLayer.__init__(self)
        #: A of shape (r,i)
        self.ball_A = BayesianBatchEnsembleLinear(
            in_features,
            config.r,
            config.batch_ensemble_size,
            bias=False,
            config=config.vbnn,
        )
        #: B of shape (o,r)
        self.ball_B = BayesianBatchEnsembleLinear(
            config.r,
            out_features,
            config.batch_ensemble_size,
            bias=False,
            config=config.vbnn,
        )
        self.scaling = config.lora_alpha / config.r
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

        # Initialize A and B
        nn.init.kaiming_uniform_(self.ball_A.weight.mu, a=math.sqrt(5))
        nn.init.zeros_(self.ball_B.weight.mu)

    def forward(self, input: torch.Tensor) -> Tensor:
        if self.config.perturbation_type == BALLConfig.PerturbationType.ADDITIVE:
            return (
                F.linear(input, self.weight, self.bias)
                + self.dropout(self.ball_B(self.ball_A(input))) * self.scaling
            )
        else:
            perturbation = self.dropout(self.ball_B(self.ball_A(input))) * self.scaling
            return F.linear(input, self.weight * (1 + perturbation), self.bias)
