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
from typing import Literal, Tuple

import torch
from bnn.nn.modules import FCGMixin, FFGMixin, InducingMixin
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
    prior_sd: float = 0.5
    """Standard deviation of the prior distribution."""
    init_sd: float = 0.1
    """Mean of the initial variational posterior distribution."""
    init_sd_sd: float = 0.01
    """Standard deviation of the initial variational posterior distribution."""
    sd_mode: Literal["softplus", "abs"] = "softplus"


class MatrixNormalPriorPosterior(nn.Module):
    M: Tensor
    prior_M: Tensor
    prior_L_u: Tensor
    prior_L_v: Tensor

    @property
    def L_u(self) -> Tensor:
        raise NotImplementedError
    
    @property
    def L_v(self) -> Tensor:
        raise NotImplementedError
    
    @property
    def U(self) -> Tensor:
        raise NotImplementedError
    
    @property
    def V(self) -> Tensor:
        raise NotImplementedError

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
        # Register the parameters and buffers.
        self.mu = nn.Parameter(torch.zeros(shape))
        self.register_buffer("prior_sigma", torch.full(shape, config.prior_sd))
        self.register_buffer("prior_mu", torch.full(shape, config.prior_mean))

        self.sigma_act_fn = None
        if config.sd_mode == "softplus":
            self.sigma_act_fn = nn.functional.softplus
            rho = inv_softplus(
                (config.init_sd + config.init_sd_sd * torch.randn(shape)).clamp(
                    min=1e-5
                )
            )
        elif config.sd_mode == "abs":
            self.sigma_act_fn = torch.abs
            rho = (config.init_sd + config.init_sd_sd * torch.randn(shape)).abs()
        else:
            raise ValueError(
                f"Invalid sd_mode {config.sd_mode}, must be 'softplus' or 'abs'"
            )
        self.rho = nn.Parameter(rho)

    def sigma(self) -> Tensor:
        """Applies softplus to the rho parameter to get the standard deviation."""
        return self.sigma_act_fn(self.rho)  # type: ignore

    def kl_divergence(self) -> Tensor:
        """Calculates the KL divergence loss between the posterior and prior distributions."""
        return self.kl_divergences().sum()

    def kl_divergences(self) -> Tensor:
        return univariate_gaussian_kl_divergence(
            self.mu, self.sigma(), self.prior_mu, self.prior_sigma
        )

    def forward(self) -> Tensor:
        """Sample from the posterior distribution using the reparameterization trick."""
        return self.mu + self.sigma() * torch.randn_like(self.mu)


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


def kl_divergence(module: nn.Module) -> Tensor | float:
    """Calculates the KL divergence loss for all variational parameters in the module.

    >>> _ = torch.manual_seed(0)
    >>> linear = VariationalLinear(10, 5)
    >>> get_model_kl_loss(linear).item()
    161.44...

    :param module: Module containing variational parameters.
    :return: Total KL divergence loss.
    """
    kl = sum(
        m.kl_divergence()  # type: ignore
        for m in module.modules()
        if hasattr(m, "kl_divergence")  # type: ignore
    )
    return kl  # type: ignore


@torch.no_grad()
def posterior_to_prior(module: nn.Module):
    for submodule in module.modules():
        if isinstance(submodule, FFGMixin):
            assert isinstance(submodule.weight_mean, Tensor)
            assert isinstance(submodule.weight_sd, Tensor)
            assert isinstance(submodule.prior_weight_mean, Tensor)
            assert isinstance(submodule.prior_weight_sd, Tensor)
            submodule.prior_weight_mean.copy_(submodule.weight_mean)
            submodule.prior_weight_sd.copy_(submodule.weight_sd)
        elif isinstance(submodule, VariationalParameter):
            assert isinstance(submodule.mu, Tensor)
            assert isinstance(submodule.rho, Tensor)
            assert isinstance(submodule.prior_mu, Tensor)
            assert isinstance(submodule.prior_sigma, Tensor)
            submodule.prior_mu.copy_(submodule.mu)
            submodule.prior_sigma.copy_(submodule.sigma())
        elif isinstance(submodule, FCGMixin):
            assert isinstance(submodule.mean, Tensor)
            assert isinstance(submodule.scale_tril, Tensor)
            assert isinstance(submodule.prior_mean, Tensor)
            assert isinstance(submodule.prior_scale_tril, Tensor)
            submodule.prior_mean.copy_(submodule.mean)
            submodule.prior_scale_tril.copy_(submodule.scale_tril)
        elif isinstance(submodule, InducingMixin):
            assert isinstance(submodule.inducing_mean, Tensor)
            assert isinstance(submodule.inducing_scale_tril, Tensor)
            pass
        elif isinstance(submodule, MatrixNormalPriorPosterior):
            assert isinstance(submodule.M, Tensor)
            assert isinstance(submodule.L_u, Tensor)
            assert isinstance(submodule.L_v, Tensor)
            assert isinstance(submodule.prior_M, Tensor)
            assert isinstance(submodule.prior_L_u, Tensor)
            assert isinstance(submodule.prior_L_v, Tensor)

            submodule.prior_M.copy_(submodule.M)
            submodule.prior_L_u.copy_(submodule.L_u)
            submodule.prior_L_v.copy_(submodule.L_v)


def replace_head(parent: nn.Module, name: str, config: VBNNConfig):
    head = parent.get_submodule(name)
    assert isinstance(head, nn.Linear), f"{head}"
    new_head = VariationalLinear(head.in_features, head.out_features, config=config)
    parent.set_submodule(name, new_head)
