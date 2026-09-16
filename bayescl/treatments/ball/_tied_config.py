from dataclasses import dataclass
from typing import ClassVar


@dataclass
class TiedBALLConfig:
    """Config for the tied variant of BALL.

    Unlike :class:`BALLConfig` this does not carry a
    :class:`~bayescl.bnn.vbnn.VBNNConfig`: the posterior is a shared
    rank-space multivariate normal rather than a per-entry mean-field Gaussian,
    so ``init_sd_sd`` and ``sd_mode`` have no analogue here. The three
    distributional knobs that do carry over are named the same.
    """

    r: int = 4
    """Rank of the LoRA adapters."""

    lora_alpha: int = 1
    """Scaling factor for the LoRA adapters."""
    dropout: float = 0.0
    """Dropout rate to use on the adapter inputs."""
    bll: bool = False
    """Whether to use Bayesian layers for the output layer."""

    prior_mean: float = 0.0
    """Mean of the prior over every column of ``A`` and every row of ``B``."""
    prior_sd: float = 0.5
    """Isotropic prior standard deviation, i.e. ``prior_L = prior_sd * I``.

    A single value covers both factors, which is forced rather than chosen: the
    shared covariance is what removes the rescaling gauge freedom, so ``A`` and
    ``B`` cannot be given separate (e.g. fan-in scaled) prior scales the way
    they are in mean-field BALL.
    """
    init_sd: float = 0.1
    """Initial standard deviation, i.e. ``L`` starts at ``init_sd * I``."""

    type: ClassVar[str] = "TiedBALL"
