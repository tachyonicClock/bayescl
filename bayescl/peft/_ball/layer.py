import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bayescl.peft._ball.config import BALLConfig
from bayescl.peft._base import AdapterBase
from bayescl.vbnn import VariationalParameter


class BALLLayer(AdapterBase):
    adapter_parameter_names = (
        "ball_A.mu",
        "ball_A.rho",
        "ball_B.mu",
        "ball_B.rho",
    )
    """Tunable parameters in BALL adapters."""


class BALLLinear(nn.Linear, BALLLayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: BALLConfig,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        BALLLayer.__init__(self)

        #: A of shape (r,i)
        self.ball_A = VariationalParameter((config.r, in_features), config.vbnn)
        #: B of shape (o,r)
        self.ball_B = VariationalParameter((out_features, config.r), config.vbnn)
        self.scaling = config.lora_alpha / config.r

        # Initialize A and B
        nn.init.kaiming_uniform_(self.ball_A.mu, a=math.sqrt(5))
        nn.init.zeros_(self.ball_B.mu)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = F.linear(input, self.weight, bias=self.bias)
        lora_A = self.ball_A.forward()
        lora_B = self.ball_B.forward()
        result += ((input @ lora_A.T) @ lora_B.T) * self.scaling
        return result
