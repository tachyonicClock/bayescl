from typing import Literal

from bayescl.base import BaseConfig


class InfLoRAConfig(BaseConfig):
    type: Literal["InfLoRA"] = "InfLoRA"
    rank: int
    threshold_start: float = 0.9
    threshold_end: float = 0.98
    max_activation_batches: int | None = 16
