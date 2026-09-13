from dataclasses import dataclass, field
from typing import ClassVar

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

    type: ClassVar[str] = "BALL"
