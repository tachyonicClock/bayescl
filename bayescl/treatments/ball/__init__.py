"""Bayesian Adaptation with Low-rank Layers (BALL).

A parameter-efficient fine-tuning method that combines LoRA-style low-rank
adapters with Bayesian variational parameters for uncertainty estimation.
"""

from ._config import BALLConfig
from ._module import BALLAdapterFactory
from ._tied_config import TiedBALLConfig
from ._tied_module import TiedBALLAdapterFactory

__all__ = [
    "BALLConfig",
    "BALLAdapterFactory",
    "TiedBALLConfig",
    "TiedBALLAdapterFactory",
]
