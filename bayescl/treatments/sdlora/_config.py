from dataclasses import dataclass
from typing import ClassVar


@dataclass
class SDLoRAConfig:
    rank_per_task: int = 1

    type: ClassVar[str] = "SDLoRA"
