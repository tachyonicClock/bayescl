"""LoRA-Ensemble: a deep ensemble of 5 independently-trained LoRA models.

Each member is a full LoRA continual learner (same recipe as `lora`, with a
different init seed) trained independently -- own optimizer, own gradients.
Predictions are combined only at evaluation time by averaging per-member
softmax probabilities.
"""

from ._arm import LoRAEnsemble
from ._module import EnsembleModule

__all__ = ["LoRAEnsemble", "EnsembleModule"]
