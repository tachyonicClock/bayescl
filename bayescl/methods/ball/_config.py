from typing import Literal

from bayescl.base import BaseConfig
from bayescl.vbnn import VBNNConfig


class BALLConfig(BaseConfig):
    type: Literal["BALL"] = "BALL"
    r: int = 4
    """Rank of the LoRA adapters."""
    lora_alpha: int = 1
    """Scaling factor for the LoRA adapters."""
    dropout: float = 0.0
    """Dropout rate to use on the adapter inputs."""
    vbnn: VBNNConfig
    """Configuration for the underlying Bayesian Neural Network."""
    bll: bool = False
    """Whether to use Bayesian layers for the output layer."""
