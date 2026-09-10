from dataclasses import dataclass
from typing import ClassVar


@dataclass
class InfLoRAConfig:
    rank: int = 10
    threshold_start: float = 0.9
    threshold_end: float = 0.98
    max_activation_batches: int | None = 16
    activation_batch_size: int = 32

    type: ClassVar[str] = "InfLoRA"
