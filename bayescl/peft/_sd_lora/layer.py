import math

import torch
from torch import nn
from torch.nn.modules.utils import _pair

from bayescl.peft._base import AdapterBase


class SDLoRA(AdapterBase):
    adapter_modules = ("A", "B")
    adapter_parameter_names = ("M",)


class SDLoRALinear(nn.Linear, SDLoRA):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank_per_task: int,
        n_tasks: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        self.A = nn.ParameterList(
            [
                nn.Parameter(torch.zeros((rank_per_task, in_features)))
                for _ in range(n_tasks)
            ]
        )
        self.B = nn.ParameterList(
            [
                nn.Parameter(torch.zeros((out_features, rank_per_task)))
                for _ in range(n_tasks)
            ]
        )
        self.M = nn.Parameter(torch.zeros((n_tasks, 1)))
        self.task: int = 0

        # Initialize the LoRA parameters
        for A_k, B_k in zip(self.A, self.B):
            nn.init.kaiming_uniform_(A_k, a=math.sqrt(5))
            nn.init.zeros_(B_k)
        nn.init.ones_(self.M)

        self.set_task(0)

    def set_task(self, task: int):
        self.task = task
        self.A.requires_grad_(False)
        self.B.requires_grad_(False)
        self.A[task].requires_grad = True
        self.B[task].requires_grad = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""

        $$
        \boldsymbol{h}^{\prime}=\left(\mathbf{W}_0+\alpha_1 \overline{\mathbf{A}_1 \mathbf{B}_1}+\alpha_2 \overline{\mathbf{A}_2 \mathbf{B}_2}+\ldots+\alpha_t \overline{\mathbf{A}_t \mathbf{B}_t}\right) \boldsymbol{x}
        $$,

        """
        h_prime = nn.functional.linear(input, self.weight, self.bias)

        for k in range(self.task):
            A_k, B_k, alpha_k = self.A[k], self.B[k], self.M[k]

            # Compute Adaptation Matrix
            W_k = B_k @ A_k  # (out_features, in_features)

            # Normalize the matrix using the Frobenius norm
            direction = W_k / (torch.norm(W_k, p="fro") + 1e-8)

            # Scale by magnitude parameter and add to output
            h_prime += alpha_k * nn.functional.linear(input, direction)

        return h_prime


class SDLoRAConv2d(nn.Conv2d, SDLoRA):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rank_per_task: int,
        n_tasks: int,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            **kwargs,
        )

        kh, kw = _pair(kernel_size)
        assert kh == kw, "Only square kernels are supported for SDLoRAConv2d"
        ks = kh
        r = rank_per_task
        groups = self.groups
        self.A = nn.ParameterList(
            [
                nn.Parameter(torch.zeros((r * ks, in_channels * ks)))
                for _ in range(n_tasks)
            ]
        )
        self.B = nn.ParameterList(
            [
                nn.Parameter(torch.zeros((out_channels // groups * ks, r * ks)))
                for _ in range(n_tasks)
            ]
        )
        self.M = nn.Parameter(torch.zeros((n_tasks, 1)))
        self.task: int = 0

        # Initialize the LoRA parameters
        for A_k, B_k in zip(self.A, self.B):
            nn.init.kaiming_uniform_(A_k, a=math.sqrt(5))
            nn.init.zeros_(B_k)
        nn.init.ones_(self.M)

        self.set_task(0)

    def set_task(self, task: int):
        self.task = task
        self.A.requires_grad_(False)
        self.B.requires_grad_(False)
        self.A[task].requires_grad = True
        self.B[task].requires_grad = True

    def conv2d(self, input: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
        return nn.functional.conv2d(
            input,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        h_prime = self.conv2d(input, self.weight, self.bias)

        for k in range(self.task + 1):
            A_k, B_k, alpha_k = self.A[k], self.B[k], self.M[k]

            # Compute Adaptation Matrix
            W_k = B_k @ A_k
            W_k = W_k.view(self.weight.shape)

            # Normalize the matrix using the Frobenius norm
            direction = W_k / (torch.norm(W_k, p="fro") + 1e-8)

            # Scale by magnitude parameter and add to output
            h_prime += alpha_k * self.conv2d(input, direction)

        return h_prime
