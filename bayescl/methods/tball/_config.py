from dataclasses import dataclass
from typing import ClassVar, Literal


@dataclass
class TBALLConfig:
    rank: int = 10
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
    bnn: Literal["FCG", "FFG", "MND"] = "FCG"
    """Bayesian core type: full covariance (FCG), fully factorized Gaussian (FFG), or matrix normal (MND)."""
    bias: bool = False
    """Whether to include bias in the Bayesian layers."""

    type: ClassVar[str] = "TBALL"

    def __post_init__(self) -> None:
        if self.bnn not in ("FCG", "FFG", "MND"):
            raise ValueError(f"bnn must be one of FCG/FFG/MND, got {self.bnn!r}")
