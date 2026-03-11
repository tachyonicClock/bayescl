from typing import Literal

from bayescl.base import BaseConfig


class CLoRAConfig(BaseConfig):
    type: Literal["CLoRA"] = "CLoRA"
    rank: int
    alpha: float
    lambda_: float
    """How much to penalize changes compared to the anchors."""
