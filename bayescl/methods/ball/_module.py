import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typeguard import typechecked

from bayescl.peft._base import AdapterBase, AdapterFactory
from bayescl.vbnn import VariationalParameter

from ._config import BALLConfig


class BALLLayer(AdapterBase):
    adapter_parameters = (
        "ball_A.mu",
        "ball_A.rho",
        "ball_B.mu",
        "ball_B.rho",
    )

def forward_lrt(x: Tensor, weight_mean: Tensor, weight_sd: Tensor) -> Tensor:
    """Local reparameterization trick forward pass."""
    mean = x @ weight_mean.T
    sd = ((x.pow(2) @ weight_sd.pow(2).T) + 1e-16).sqrt()
    return mean + torch.randn_like(mean) * sd


def forward_flipout(x: Tensor, weight_mean: Tensor, weight_sd: Tensor) -> Tensor:
    """Flipout forward pass."""
    delta_weight = weight_sd * torch.randn_like(weight_sd)
    z_mean = x @ weight_mean.T
    sign_input = x.clone().uniform_(-1, 1).sign()
    sign_output = z_mean.clone().uniform_(-1, 1).sign()
    perturbed_outputs = ((x * sign_input) @ delta_weight.T) * sign_output
    return z_mean + perturbed_outputs


class BALLLinear(nn.Linear, BALLLayer):
    def __init__(self, in_features, out_features, config: BALLConfig, **kwargs):
        self.config = config
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        BALLLayer.__init__(self)
        self.ball_A = VariationalParameter((config.r, in_features), config.vbnn)
        self.ball_B = VariationalParameter((out_features, config.r), config.vbnn)
        self.scaling = config.lora_alpha / config.r
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.ball_A.mu, a=math.sqrt(5))
        nn.init.zeros_(self.ball_B.mu)

    def forward_lrt(self, input):
        bottleneck = forward_lrt(input, self.ball_A.mu, self.ball_A.sigma())
        z = forward_lrt(bottleneck, self.ball_B.mu, self.ball_B.sigma())
        return F.linear(input, self.weight, self.bias) + z * self.scaling

    def forward_flipout(self, input):
        bottleneck = forward_flipout(input, self.ball_A.mu, self.ball_A.sigma())
        z = forward_flipout(bottleneck, self.ball_B.mu, self.ball_B.sigma())
        return F.linear(input, self.weight, self.bias) + z * self.scaling

    def forward_none(self, input):
        lora_A = self.ball_A.forward()
        lora_B = self.ball_B.forward()
        z = (input @ lora_A.T) @ lora_B.T
        return F.linear(input, self.weight, self.bias) + z * self.scaling

    def forward(self, input):
        input = self.dropout(input)
        if not self.training:
            return self.forward_none(input)
        return self.forward_none(input)


class BALLConv2d(BALLLayer, nn.Conv2d):
    """A Bayesian Adaptation Layer using the BALL method for Conv2d layers."""

    def __init__(self, in_channels, out_channels, kernel_size, config: BALLConfig, **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        BALLLayer.__init__(self)
        kh, kw = _pair(kernel_size)
        assert kh == kw, "Only square kernels are supported for BALLConv2d"
        ks = kh
        groups = self.groups
        rank_ks = config.r * ks
        self.ball_A = VariationalParameter((rank_ks, in_channels * ks), config.vbnn)
        self.ball_B = VariationalParameter((out_channels // groups * ks, rank_ks), config.vbnn)
        self.scaling = config.lora_alpha / config.r

    def forward(self, x: Tensor) -> Tensor:
        weight_delta = self.ball_B.forward() @ self.ball_A.forward()
        weight_delta = weight_delta * self.scaling
        weight = self.weight + weight_delta.view_as(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class BALLAdapterFactory(AdapterFactory):
    """A factory for creating BALL adapters."""

    @typechecked
    def __init__(self, config: BALLConfig) -> None:
        self.config = config

    def _get_replacement(self, module: nn.Module) -> nn.Module:
        if isinstance(module, nn.Linear):
            return BALLLinear(
                module.in_features, module.out_features, config=self.config
            )
        elif isinstance(module, nn.Conv2d):
            return BALLConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,  # type: ignore[arg-type]
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                config=self.config,
            )
        else:
            raise ValueError(f"Unsupported layer type: {type(module)}")

    def __call__(self, module: nn.Module) -> nn.Module:
        """Create an adapter for a given module."""
        replacement = self._get_replacement(module)
        if isinstance(replacement, nn.Module):
            replacement.load_state_dict(module.state_dict(), strict=False)
        return replacement
