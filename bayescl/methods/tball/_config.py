from typing import Literal

from bayescl.base import BaseConfig


class TBALLConfig(BaseConfig):
    type: Literal["TBALL"] = "TBALL"
    rank: int
    """Rank of the TBALL adapters."""
    alpha: float = 1.0
    """Scaling factor for the TBALL adapters."""
    prior_mean: float = 0.0
    """Prior mean for the Bayesian layers."""
    prior_weight_sd: float = 1.0
    """Prior standard deviation for the weights."""
    init_sd: float = 1e-4
    """Standard deviation for initializing the variational parameters."""
    nonlinearity_scale: float = 1.0
    """Scale for the nonlinearity in the Bayesian layers."""
    bnn: Literal["FCG", "FFG"] = "FCG"
    """Full covariance (FCG) or fully factorized Gaussian (FFG) Bayesian layers."""
    bias: bool = False
    """Whether to include bias in the Bayesian layers."""
