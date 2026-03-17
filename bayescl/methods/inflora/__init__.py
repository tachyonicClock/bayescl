"""InfLoRA (Interference-Free Low-Rank Adaptation).

This **submodule** `bayescl/methods/inflora` was generated using VS Code Agent Mode
using the original paper and code base as context. I have reviewed the generated code
and am satisfied that it captures the proposed method.

Liang, Y.-S., & Li, W.-J. (2024). InfLoRA: Interference-Free Low-Rank Adaptation for
Continual Learning. 23638–23647.
https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html
"""

from ._config import InfLoRAConfig
from ._module import InfLoRAAdapterFactory
from ._plugin import InfLoRAPlugin

__all__ = ["InfLoRAConfig", "InfLoRAPlugin", "InfLoRAAdapterFactory"]
