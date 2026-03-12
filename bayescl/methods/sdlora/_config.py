from typing import Literal

from bayescl.base import BaseConfig


class SDLoRAConfig(BaseConfig):
    type: Literal["SDLoRA"] = "SDLoRA"
    rank_per_task: int
