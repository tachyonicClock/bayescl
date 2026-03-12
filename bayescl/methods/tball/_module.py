from typing import Tuple

import torch
import torch.nn as nn
from bnn.nn.modules import FCGLinear, FCGMixin, FFGLinear, FFGMixin
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typeguard import typechecked

from bayescl.peft._base import AdapterBase, AdapterFactory

from ._config import TBALLConfig


class WeightModule(nn.Module):
    def __init__(self, shape: Tuple[int, int]):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(*shape))
        self.bias = None


class FFGParameter(FFGMixin, WeightModule):
    def __init__(
        self,
        shape: Tuple[int, int],
        config: TBALLConfig,
    ):
        super().__init__(
            shape,
            prior_mean=config.prior_mean,
            prior_weight_sd=config.prior_weight_sd,
            init_sd=config.init_sd,
            nonlinearity_scale=config.nonlinearity_scale,
        )

    def sample(self) -> Tensor:
        return self.weight_mean + self.weight_sd * torch.randn_like(self.weight_mean)


class FCGParameter(FCGMixin, WeightModule):
    def __init__(
        self,
        shape: Tuple[int, int],
        config: TBALLConfig,
    ):
        super().__init__(
            shape,
            prior_mean=config.prior_mean,
            prior_weight_sd=config.prior_weight_sd,
            init_sd=config.init_sd,
            nonlinearity_scale=config.nonlinearity_scale,
        )

    def sample(self) -> Tensor:
        sample = self.mean + self.scale_tril @ torch.randn_like(self.mean)
        return sample.view_as(self.weight)


class BALLLayer(AdapterBase):
    pass


class TBALLLinear(BALLLayer, nn.Linear):
    """A Bayesian Adaptation Layer using the TBALL method for Linear layers."""

    project_down: Tensor
    """Random projection matrix to reduce input dimensionality."""
    bayes_core: FCGLinear | FFGLinear
    """The Bayesian core layer."""
    project_up: Tensor
    """Random projection matrix to restore output dimensionality."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: TBALLConfig,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        BALLLayer.__init__(self)

        self._scaling = config.alpha / config.rank

        # Random projection matrices
        project_down = torch.empty(in_features, config.rank)
        project_up = torch.empty(config.rank, out_features)
        # Use orthogonal initialization because it preserves norms
        nn.init.orthogonal_(project_down)
        nn.init.orthogonal_(project_up)
        self.register_buffer("project_down", project_down)
        self.register_buffer("project_up", project_up)

        # Bayesian core layer
        if config.bnn == "FCG":
            self.adapter_parameters = (
                "bayes_core._scale_diag",
                "bayes_core._scale_tril",
                "bayes_core.mean",
            )
            self.bayes_core = FCGLinear(
                config.rank,
                config.rank,
                bias=config.bias,
                prior_mean=config.prior_mean,
                prior_weight_sd=config.prior_weight_sd,
                init_sd=config.init_sd,
                nonlinearity_scale=config.nonlinearity_scale,
            )
        elif config.bnn == "FFG":
            self.adapter_parameters = (
                "bayes_core.weight_mean",
                "bayes_core._weight_sd",
            )
            if config.bias:
                self.adapter_parameters = (
                    "bayes_core.bias_mean",
                    "bayes_core._bias_sd",
                    *self.adapter_parameters,
                )
            self.bayes_core = FFGLinear(
                config.rank,
                config.rank,
                bias=config.bias,
                prior_mean=config.prior_mean,
                prior_weight_sd=config.prior_weight_sd,
                init_sd=config.init_sd,
                nonlinearity_scale=config.nonlinearity_scale,
            )
        else:
            raise ValueError(f"Unsupported BNN type: {config.bnn}")

    def forward(self, x: Tensor) -> Tensor:
        z = x @ self.project_down  # (batch_size, rank)
        z = self.bayes_core(z)  # (batch_size, rank)
        z = z @ self.project_up  # (batch_size, out_features)
        return F.linear(x, self.weight, self.bias) + z * self._scaling


class TBALLConv2d(BALLLayer, nn.Conv2d):
    """A Bayesian Adaptation Layer using the TBALL method for Conv2d layers."""

    tball_A: Tensor
    """Random projection matrix to reduce input dimensionality."""
    tball_C: Tensor
    """Random projection matrix to restore output dimensionality."""
    tball_B: FFGParameter | FCGParameter
    """The Bayesian core layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        config: TBALLConfig,
        **kwargs,
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        BALLLayer.__init__(self)
        if config.bias:
            raise NotImplementedError("TBALLConv2d does not support bias yet")

        self._scaling = config.alpha / config.rank

        # Random projection matrices
        kh, kw = _pair(kernel_size)
        assert kh == kw, "Only square kernels are supported for TBALLConv2d"
        ks = kh
        groups = self.groups
        rank_ks = config.rank * ks
        tball_A = torch.empty(rank_ks, in_channels * ks)  # A
        tball_C = torch.empty(out_channels // groups * ks, rank_ks)  # B

        # Use orthogonal initialization because it preserves norms
        nn.init.orthogonal_(tball_A)
        nn.init.orthogonal_(tball_C)
        self.tball_A = nn.Parameter(tball_A, requires_grad=False)
        self.tball_C = nn.Parameter(tball_C, requires_grad=False)

        # Bayesian core layer
        if config.bnn == "FCG":
            self.adapter_parameters = (
                "tball_B._scale_diag",
                "tball_B._scale_tril",
                "tball_B.mean",
            )
            self.tball_B = FCGParameter((rank_ks, rank_ks), config)
        elif config.bnn == "FFG":
            self.adapter_parameters = (
                "tball_B.weight_mean",
                "tball_B._weight_sd",
            )
            self.tball_B = FFGParameter((rank_ks, rank_ks), config)
        else:
            raise ValueError(f"Unsupported BNN type: {config.bnn}")

    def forward(self, x: Tensor) -> Tensor:
        weight_delta = self.tball_C @ self.tball_B.sample() @ self.tball_A
        weight = self.weight + weight_delta.view_as(self.weight) * self._scaling
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class TBALLAdapterFactory(AdapterFactory):
    """A factory for creating TBALL adapters."""

    @typechecked
    def __init__(self, config: TBALLConfig) -> None:
        self.config = config

    def _get_replacement(self, module: nn.Module) -> nn.Module:
        if isinstance(module, nn.Linear):
            return TBALLLinear(
                module.in_features, module.out_features, config=self.config
            )
        elif isinstance(module, nn.Conv2d):
            return TBALLConv2d(
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
