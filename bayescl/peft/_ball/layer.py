import math

import torch
import torch.nn as nn
from torch import Tensor

from bayescl.peft._ball.config import BALLConfig, VarianceReduction
from bayescl.peft._base import AdapterBase
from bayescl.vbnn import VariationalParameter


class BALLLayer(AdapterBase):
    adapter_parameter_names = (
        "ball_A.mu",
        "ball_A.rho",
        "ball_B.mu",
        "ball_B.rho",
    )
    """Tunable parameters in BALL adapters."""


def forward_ball_lrt(
    x: Tensor, A_mu: Tensor, A_sigma: Tensor, B_mu: Tensor, B_sigma: Tensor
) -> Tensor:
    """Local reparameterization trick forward pass.

    Takes advantage of the fact that a linear transformation of a Gaussian is also
    Gaussian. Instead of sampling weights and then doing the forward pass, we compute
    the mean and standard deviation of the output distribution and sample from that.

    Based on:
    * https://github.com/ThirstyScholar/bayes-by-backprop/blob/06d9c714c3e429cfc23e629fcd0992a57412200a/BBB/BNNLayer.py#L42
    * https://github.com/microsoft/bayesianize/blob/6aab95e1305d8ed62a84111dd61e53113a850ecc/bnn/nn/mixins/variational/ffg.py#L138

    Kingma, D. P., Salimans, T., & Welling, M. (2015). Variational dropout and the local
    reparameterization trick. https://arxiv.org/abs/1506.02557
    """
    # (batch_size, out_features)
    z_mean = (x @ A_mu.T) @ B_mu.T
    # (batch_size, out_features)
    z_sd = ((x.square() @ A_sigma.square().T) @ B_sigma.square().T).sqrt()
    z = z_mean + z_sd * torch.randn_like(z_mean)
    return z


def forward_flipout(x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """Flipout forward pass.

    Based on:
    * https://github.com/IntelLabs/bayesian-torch/blob/dfb44c29ec9544589097be31a892214520ddb2a4/bayesian_torch/layers/flipout_layers/linear_flipout.py#L145
    """
    delta_weight = sigma * torch.randn_like(sigma)
    z_mean = x @ mu.T
    sign_input = x.clone().uniform_(-1, 1).sign()
    sign_output = z_mean.clone().uniform_(-1, 1).sign()
    perturbed_outputs = ((x * sign_input) @ delta_weight.T) * sign_output
    return z_mean + perturbed_outputs


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
        self.ball_A = VariationalParameter((config.r, in_features), config.vbnn)
        #: B of shape (o,r)
        self.ball_B = VariationalParameter((out_features, config.r), config.vbnn)
        self.scaling = config.lora_alpha / config.r

        # Initialize A and B
        nn.init.kaiming_uniform_(self.ball_A.mu, a=math.sqrt(5))
        nn.init.zeros_(self.ball_B.mu)

    def forward_lrt(self, input: torch.Tensor) -> torch.Tensor:
        z = forward_ball_lrt(
            input,
            A_mu=self.ball_A.mu,
            A_sigma=self.ball_A.sigma(),
            B_mu=self.ball_B.mu,
            B_sigma=self.ball_B.sigma(),
        )
        return super().forward(input) + z * self.scaling

    def forward_flipout(self, input: torch.Tensor) -> torch.Tensor:
        bottleneck = forward_flipout(input, self.ball_A.mu, self.ball_A.sigma())
        z = forward_flipout(bottleneck, self.ball_B.mu, self.ball_B.sigma())
        return super().forward(input) + z * self.scaling

    def forward_none(self, input: torch.Tensor) -> torch.Tensor:
        lora_A = self.ball_A.forward()
        lora_B = self.ball_B.forward()
        z = (input @ lora_A.T) @ lora_B.T
        return super().forward(input) + z * self.scaling

    def forward(self, input: torch.Tensor) -> Tensor:
        # If not training, do a standard forward pass
        if not self.training:
            return self.forward_none(input)

        # Training mode: use the selected variance reduction method
        if self.config.mode == VarianceReduction.LRT:
            return self.forward_lrt(input)
        elif self.config.mode == VarianceReduction.FLIPOUT:
            return self.forward_flipout(input)
        else:
            return self.forward_none(input)
