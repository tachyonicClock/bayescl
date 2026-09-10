from dataclasses import dataclass
from typing import ClassVar


@dataclass
class CLoRAConfig:
    rank: int = 10
    """Rank of LoRA adapters."""
    alpha: float = 1.0
    """LoRA scaling factor."""
    lambda_: float = 1.0
    """How much to penalize changes compared to the anchors."""

    type: ClassVar[str] = "CLoRA"
