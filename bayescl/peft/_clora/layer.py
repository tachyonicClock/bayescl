import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bayescl.peft._base import AdapterBase
from bayescl.peft._clora.config import CLoRAConfig


class CLoRALayer(AdapterBase):
    #: LoRA A parameter of shape (r, in_features)
    clora_A: nn.Parameter
    #: LoRA B parameter of shape (out_features, r)
    clora_B: nn.Parameter
    #: Frozen LoRA A parameter of shape (r, in_features) from previous tasks
    anchor_A: torch.Tensor
    #: Frozen LoRA B parameter of shape (out_features, r) from previous tasks
    anchor_B: torch.Tensor

    adapter_parameter_names = ("clora_A", "clora_B")

    def __init__(self, config: CLoRAConfig):
        if not isinstance(config, CLoRAConfig):
            raise ValueError(
                f"Expected config to be of type ConfigCLoRA, got {type(config)}"
            )
        #: Configuration for the BALL adapter.
        self.config = config

    @property
    def rank(self) -> int:
        return self.config.r

    def reset_clora(self) -> None:
        pass


class CLoRALinear(nn.Linear, CLoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: CLoRAConfig,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        CLoRALayer.__init__(self, config)

        self.clora_A = nn.Parameter(torch.empty(self.rank, in_features))
        self.clora_B = nn.Parameter(torch.empty(out_features, self.rank))
        self.anchor_A = nn.Buffer(torch.zeros(self.rank, in_features))
        self.anchor_B = nn.Buffer(torch.zeros(out_features, self.rank))
        self.scaling = self.config.lora_alpha / self.rank
        self.reset_clora()

    def reset_clora(self) -> None:
        nn.init.kaiming_uniform_(self.clora_A, a=math.sqrt(5))
        nn.init.zeros_(self.clora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight + self.scaling * self.anchor_B @ self.anchor_A
        result = F.linear(input, weight, bias=self.bias)
        result += (
            input @ self.clora_A.transpose(0, 1) @ self.clora_B.transpose(0, 1)
        ) * self.scaling
        return result
