"""Scalable Decoupled LoRA (SD-LoRA).

Wu, Y., Piao, H., Huang, L.-K., Wang, R., Li, W., Pfister, H., Meng, D., Ma, K., & Wei,
Y. (2025). SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental
Learning (arXiv:2501.13198). arXiv. https://doi.org/10.48550/arXiv.2501.13198
"""

from ._config import SDLoRAConfig
from ._module import SDLoRAAdapterFactory
from ._plugin import SDLoRAPlugin

__all__ = ["SDLoRAConfig", "SDLoRAPlugin", "SDLoRAAdapterFactory"]
