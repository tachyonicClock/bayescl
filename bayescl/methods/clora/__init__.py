"""Continual LoRA (c-LoRA).

Smith, J. S., Hsu, Y.-C., Zhang, L., Hua, T., Kira, Z., Shen, Y., & Jin, H. (2024).
Continual diffusion: Continual customization of text-to-image diffusion with c-LoRA.
2024. https://openreview.net/forum?id=TZdEgwZ6f3
"""

from ._config import CLoRAConfig
from ._module import CLoRAAdapterFactory
from ._plugin import CLoRAPlugin

__all__ = ["CLoRAConfig", "CLoRAPlugin", "CLoRAAdapterFactory"]
