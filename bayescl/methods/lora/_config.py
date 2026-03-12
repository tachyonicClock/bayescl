from typing import Literal

from bayescl.base import BaseConfig


class LoRAConfig(BaseConfig):
    type: Literal["LoRA"] = "LoRA"
    r: int = 16
    lora_alpha: int = 1
    lora_dropout: float = 0.0
