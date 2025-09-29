import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from bnn.nn.modules import FFGLinear, FFGMixin
from torch import Tensor

from bayescl.peft._ball.config import BALLConfig
from bayescl.peft._base import AdapterBase


@dataclass
class FFGState:
    weight_mean: Tensor
    weight_sd: Tensor
    bias_mean: Optional[Tensor]
    bias_sd: Optional[Tensor]


@torch.no_grad()
def get_FFG_posterior(ffg: FFGMixin) -> FFGState:
    return FFGState(
        weight_mean=ffg.weight_mean,
        weight_sd=ffg.weight_sd,
        bias_mean=ffg.bias_mean,
        bias_sd=ffg.bias_sd,
    )


@torch.no_grad()
def set_FFG_prior(ffg: FFGMixin, state: FFGState):
    assert isinstance(ffg.prior_weight_mean, Tensor)
    assert isinstance(ffg.prior_weight_sd, Tensor)
    ffg.prior_weight_mean.copy_(state.weight_mean)
    ffg.prior_weight_sd.copy_(state.weight_sd)

    if ffg.has_bias and state.bias_mean is not None and state.bias_sd is not None:
        assert isinstance(ffg.prior_bias_mean, Tensor)
        assert isinstance(ffg.prior_bias_sd, Tensor)
        ffg.prior_bias_mean.copy_(state.bias_mean)
        ffg.prior_bias_sd.copy_(state.bias_sd)
    elif ffg.has_bias:
        raise ValueError("bias_mean and bias_sd must be provided if FFG has bias")


def posterior_to_prior(module: nn.Module):
    for submodule in module.modules():
        if isinstance(submodule, FFGMixin):
            set_FFG_prior(submodule, get_FFG_posterior(submodule))


class BALLLayer(AdapterBase):
    adapter_parameter_names = (
        "ball_A.weight_mean",
        "ball_A._weight_sd",
        "ball_B.weight_mean",
        "ball_B._weight_sd",
    )
    """Tunable parameters in BALL adapters."""

    def kl_divergence(self) -> torch.Tensor:
        raise NotImplementedError()


class BALLLinear(nn.Linear, BALLLayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: BALLConfig,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        BALLLayer.__init__(self)

        #: A of shape (r,i)
        self.ball_A = FFGLinear(
            in_features=in_features,
            out_features=config.r,
            bias=False,
            prior_mean=config.prior_mean,
            prior_weight_sd=config.prior_weight_sd,
            prior_bias_sd=config.prior_bias_sd,
            init_sd=config.init_sd,
            max_sd=config.max_sd,
            local_reparameterization=config.local_reparameterization,
            nonlinearity_scale=config.nonlinearity_scale,
            sqrt_width_scaling=config.sqrt_width_scaling,
        )
        #: B of shape (o,r)
        self.ball_B = FFGLinear(
            in_features=config.r,
            out_features=out_features,
            bias=False,
            prior_mean=config.prior_mean,
            prior_weight_sd=config.prior_weight_sd,
            prior_bias_sd=config.prior_bias_sd,
            init_sd=config.init_sd,
            max_sd=config.max_sd,
            local_reparameterization=config.local_reparameterization,
            nonlinearity_scale=config.nonlinearity_scale,
            sqrt_width_scaling=config.sqrt_width_scaling,
        )
        self.scaling = config.lora_alpha / config.r

        # Initialize A and B
        nn.init.kaiming_uniform_(self.ball_A.weight_mean, a=math.sqrt(5))
        nn.init.zeros_(self.ball_B.weight_mean)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input) + self.ball_B(self.ball_A(input)) * self.scaling
