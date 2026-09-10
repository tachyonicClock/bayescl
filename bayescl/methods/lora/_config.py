from dataclasses import dataclass
from typing import ClassVar


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 1
    lora_dropout: float = 0.0

    type: ClassVar[str] = "LoRA"
