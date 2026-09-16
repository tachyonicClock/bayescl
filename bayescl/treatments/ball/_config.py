from dataclasses import dataclass, field
from typing import ClassVar, Literal

from bayescl.bnn.vbnn import VBNNConfig


@dataclass
class BALLConfig:
    r: int = 4
    """Rank of the LoRA adapters."""

    lora_alpha: int = 1
    """Scaling factor for the LoRA adapters."""
    dropout: float = 0.0
    """Dropout rate to use on the adapter inputs."""
    vbnn: VBNNConfig = field(default_factory=VBNNConfig)
    """Configuration for the underlying Bayesian Neural Network."""
    bll: bool = False
    """Whether to use Bayesian layers for the output layer."""
    sampling: Literal["weight", "lrt", "flipout"] = "weight"
    """Variance reduction estimator used for the training-time forward pass.

    ``weight`` takes a single weight draw and shares it across the batch; the
    other two give each row its own implicit draw. Named ``weight`` rather than
    ``none`` because the layer is stochastic either way -- what varies is how
    many draws a forward pass takes, not whether it samples at all.
    """

    type: ClassVar[str] = "BALL"
