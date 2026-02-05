import torch
import torch.nn as nn
from bnn.nn.modules import FCGLinear, FFGLinear
from torch import Tensor
from torch.nn import functional as F

from bayescl.config import TBALLConfig
from bayescl.peft._base import AdapterBase


class BALLLayer(AdapterBase):
    adapter_parameter_names = None
    """Tunable parameters in BALL adapters."""


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
            self.adapter_parameter_names = (
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
            self.adapter_parameter_names = [
                "bayes_core.weight_mean",
                "bayes_core._weight_sd",
            ]
            if config.bias:
                self.adapter_parameter_names += [
                    "bayes_core.bias_mean",
                    "bayes_core._bias_sd",
                ]
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
