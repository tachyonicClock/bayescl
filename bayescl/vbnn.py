"""Variational Bayesian Neural Networks (VBNNs) [Blundell2015]_.

This module assumes that both the prior and posterior distributions for each
weight is the univariate Gaussian distribution.

I found the following resources helpful for understanding and implementing VBNNs:

#. `Bayesianize: A Bayesian neural network wrapper in pytorch <https://github.com/microsoft/bayesianize/tree/main>`_

..  [Blundell2015] Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight uncertainty in neural network. In F. R.
    Bach & D. M. Blei (Eds.), Proceedings of the 32nd international conference on machine learning, ICML 2015, lille,
    france, 6-11 july 2015 (Vol. 37, pp. 1613–1622). JMLR.org. http://proceedings.mlr.press/v37/blundell15.html
"""

import math
from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import torch
from bnn.nn.modules import FFGLinear, FFGMixin
from torch import Tensor, nn


def univariate_gaussian_kl_divergence(
    mu_0: Tensor, std_0: Tensor, mu_1: Tensor, std_1: Tensor
) -> Tensor:
    r"""
    Batch calculates kl divergence between two vectors of univariate gaussians (p_0 || p_1).

    .. math::

        D_{\text{KL}}\left({\mathcal {p}}_0\parallel {\mathcal {p}}_1\right)=
            \log {\frac {\sigma _{1}}{\sigma _{0}}}
            +{\frac {\sigma _{0}^{2}+{\left(\mu _{0}-\mu _{1}\right)}^{2}}{2\sigma _{1}^{2}}}
            -{\frac {1}{2}}

    * https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions

    :param mu_0: Batch or scalar of means defining the first distributions.
    :param std_0: Batch or scalar of standard deviations defining the first distributions.
    :param mu_1: Batch or scalar of means defining the second distributions.
    :param std_1: Batch or scalar of standard deviations defining the second distributions.
    :return: Batch of KL divergence values between the two distributions.
    """
    return (
        torch.log(std_1)
        - torch.log(std_0)
        + (std_0**2 + (mu_0 - mu_1) ** 2) / (2 * (std_1**2))
        - 0.5
    )


def inv_softplus(x: Tensor) -> Tensor:
    """Reverse of the softplus function.

    >>> nn.functional.softplus(inv_softplus(torch.tensor(1e-4)))
    tensor(0.0001)
    """
    return torch.log(torch.expm1(x))


@dataclass
class VBNNConfig:
    """Config for Variational Bayesian Neural Networks."""

    prior_mean: float = 0.0
    """Mean of the prior distribution."""
    prior_weight_sd: float = 0.5
    """Standard deviation of the prior distribution."""
    init_sd: float = 0.1
    """Mean of the initial variational posterior distribution."""
    init_sd_sd: float = 0.1
    """Standard deviation of the initial variational posterior distribution."""
    deterministic_eval: bool = False
    """If True, in evaluation mode, the forward pass will not add noise to the output."""


class VariationalParameter(nn.Module):
    """Implements a Gaussian variational parameter using the reparameterization trick.

    The owner of this object is responsible for initializing ``mu``.
    """

    #: Mean of the posterior distribution.
    mu: nn.Parameter
    #: Parameterization of the standard deviation of the posterior distribution.
    #: calculated as :math:`\sigma = \text{softplus}(\rho)`.
    rho: nn.Parameter
    #: Mean of the prior distribution.
    prior_mu: Tensor
    #: Standard deviation of the prior distribution.
    prior_sigma: Tensor

    def __init__(
        self,
        shape: Tuple[int, ...],
        config: VBNNConfig,
    ) -> None:
        """Constructs a variational parameter.

        :param shape: Shape of the parameter.
        :param config: Configuration for the variational parameter.
        """
        super().__init__()
        self._deterministic_eval = config.deterministic_eval

        # Register the parameters and buffers.
        rho_mean = inv_softplus(
            (config.init_sd + config.init_sd_sd * torch.randn(shape)).clamp(min=1e-5)
        )
        self.mu = nn.Parameter(torch.zeros(shape))
        self.rho = nn.Parameter(rho_mean)
        self.register_buffer("prior_sigma", torch.full(shape, config.prior_weight_sd))
        self.register_buffer("prior_mu", torch.full(shape, config.prior_mean))

    def weight_sd(self) -> Tensor:
        """Applies softplus to the rho parameter to get the standard deviation."""
        return nn.functional.softplus(self.rho)

    def kl_divergence(self) -> Tensor:
        """Calculates the KL divergence loss between the posterior and prior distributions."""
        return self.kl_divergences().sum()

    def kl_divergences(self) -> Tensor:
        return univariate_gaussian_kl_divergence(
            self.mu, self.weight_sd(), self.prior_mu, self.prior_sigma
        )

    def forward(self) -> Tensor:
        """Sample from the posterior distribution using the reparameterization trick."""
        if self.training or not self._deterministic_eval:
            return self.mu + self.weight_sd() * torch.randn_like(self.mu)
        return self.mu


class VariationalLinear(nn.Module):
    """Linear layer with variational parameters.

    >>> VariationalLinear(2, 2)
    VariationalLinear(
      (weight): VariationalParameter()
      (bias): VariationalParameter()
    )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: VBNNConfig = VBNNConfig(),
    ):
        super().__init__()
        self.dim_in = in_features
        self.dim_out = out_features

        self.weight = VariationalParameter(
            shape=(out_features, in_features), config=config
        )

        if bias:
            self.bias = VariationalParameter(shape=(out_features,), config=config)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize weights
        # Copy paste of PyTorch default initialization for nn.Linear
        nn.init.kaiming_uniform_(self.weight.mu, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.mu)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias.mu, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of linear layer with variational parameters."""
        weight = self.weight.forward()
        bias = self.bias.forward() if self.bias is not None else None
        return nn.functional.linear(x, weight, bias)


def iterate_variational_parameters(
    module: nn.Module,
) -> Generator[Tuple[str, VariationalParameter], None, None]:
    for name, submodule in module.named_modules():
        if isinstance(submodule, VariationalParameter):
            yield name, submodule


def kl_divergence(module: nn.Module) -> Tensor:
    """Calculates the KL divergence loss for all variational parameters in the module.

    >>> _ = torch.manual_seed(0)
    >>> linear = VariationalLinear(10, 5)
    >>> get_model_kl_loss(linear).item()
    161.44...

    :param module: Module containing variational parameters.
    :return: Total KL divergence loss.
    """
    kl = sum(
        m.kl_divergence()
        for m in module.modules()
        if hasattr(m, "kl_divergence")  # type: ignore
    )
    assert isinstance(kl, Tensor)
    return kl


@dataclass
class FFGState:
    weight_mean: Tensor
    weight_sd: Tensor
    bias_mean: Optional[Tensor]
    bias_sd: Optional[Tensor]


@torch.no_grad()
def get_posterior(vb_layer: FFGMixin | VariationalParameter) -> FFGState:
    if isinstance(vb_layer, VariationalParameter):
        return FFGState(
            weight_mean=vb_layer.mu,
            weight_sd=vb_layer.weight_sd(),
            bias_mean=None,
            bias_sd=None,
        )
    elif isinstance(vb_layer, FFGMixin):
        return FFGState(
            weight_mean=vb_layer.weight_mean,
            weight_sd=vb_layer.weight_sd,
            bias_mean=vb_layer.bias_mean,
            bias_sd=vb_layer.bias_sd,
        )
    else:
        raise TypeError(
            f"`vb_layer` must be {FFGLinear} or {VariationalParameter}, got {type(vb_layer)}"
        )


@torch.no_grad()
def set_prior(vb_layer: FFGMixin | VariationalParameter, state: FFGState):
    if isinstance(vb_layer, VariationalParameter):
        vb_layer.prior_mu = state.weight_mean
        vb_layer.prior_sigma = state.weight_sd
    elif isinstance(vb_layer, FFGMixin):
        assert isinstance(vb_layer.prior_weight_mean, Tensor)
        assert isinstance(vb_layer.prior_weight_sd, Tensor)
        vb_layer.prior_weight_mean.copy_(state.weight_mean)
        vb_layer.prior_weight_sd.copy_(state.weight_sd)

        if (
            vb_layer.has_bias
            and state.bias_mean is not None
            and state.bias_sd is not None
        ):
            assert isinstance(vb_layer.prior_bias_mean, Tensor)
            assert isinstance(vb_layer.prior_bias_sd, Tensor)
            vb_layer.prior_bias_mean.copy_(state.bias_mean)
            vb_layer.prior_bias_sd.copy_(state.bias_sd)
        elif vb_layer.has_bias:
            raise ValueError("bias_mean and bias_sd must be provided if FFG has bias")
    else:
        raise TypeError("ffg must be FFGMixin or VariationalParameter")


def posterior_to_prior(module: nn.Module):
    for submodule in module.modules():
        if isinstance(submodule, FFGMixin):
            set_prior(submodule, get_posterior(submodule))
        if isinstance(submodule, VariationalParameter):
            set_prior(submodule, get_posterior(submodule))
