import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bayescl import vbnn
from bayescl.peft._ball.config import BALLConfig
from bayescl.peft._base import AdapterBase


class BALLLayer(AdapterBase):
    adapter_parameter_names = ("ball_A.mu", "ball_A.rho", "ball_B.mu", "ball_B.rho")

    def __init__(self, config: BALLConfig):
        if not isinstance(config, BALLConfig):
            raise ValueError(
                f"Expected config to be of type BALLConfig, got {type(config)}"
            )
        #: Configuration for the BALL adapter.
        self.config = config


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
        BALLLayer.__init__(self, config)

        #: A of shape (r,i)
        self.ball_A = vbnn.VariationalParameter(
            (self.config.r, in_features), self.config.vbnn
        )
        #: B of shape (o,r)
        self.ball_B = vbnn.VariationalParameter(
            (out_features, self.config.r), self.config.vbnn
        )
        self.scaling = self.config.lora_alpha / self.config.r

        nn.init.kaiming_uniform_(self.ball_A.mu, a=math.sqrt(5))
        nn.init.zeros_(self.ball_B.mu)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = F.linear(input, self.weight, bias=self.bias)
        lora_A = self.ball_A.forward()
        lora_B = self.ball_B.forward()
        result += (
            input @ lora_A.transpose(0, 1) @ lora_B.transpose(0, 1)
        ) * self.scaling
        return result
