import math

import torch
from torch import Tensor, nn

from bayescl.peft import AdapterBase, AdapterFactory, iter_named_adapters

from ._config import CLoRAConfig


class CLoRAModule(AdapterBase):
    adapter_parameters = ("A", "B")
    adapter_buffers = ("anchor",)

    A: nn.Parameter
    B: nn.Parameter
    anchor: Tensor

    def reset_adapter(self) -> None:
        pass

    def self_regularization_loss(self) -> Tensor:
        r"""Penalize changes to features learned by previous tasks.

        .. math::

            \mathcal{L}_{\text{reg}} =
            || \left| \sum_{t'=1}^{t-1} \mathbf{B}_{t'} \mathbf{A}_{t'} \right|
            \odot \mathbf{B}_t \mathbf{A}_t ||_F^2

        where :math:`\sum_{t'=1}^{t-1} \mathbf{B}_{t'} \mathbf{A}_{t'}` is pre-computed
        and stored in `anchor` to save computation.
        """
        return (self.anchor.abs() * (self.B @ self.A)).norm(p="fro") ** 2


class CLoRAConv2d(nn.Conv2d, CLoRAModule):
    def __init__(
        self,
        config: CLoRAConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        **kwargs,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        r = config.rank
        assert self.kernel_size[0] == self.kernel_size[1], (
            "Only square kernels are supported"
        )
        ks = self.kernel_size[0]

        shape_A = (r * ks, in_channels * ks)
        shape_B = (out_channels // self.groups * ks, r * ks)
        self.A = nn.Parameter(torch.empty(shape_A))
        self.B = nn.Parameter(torch.empty(shape_B))
        self.anchor = nn.Buffer(torch.zeros(shape_B[0], shape_A[1]))
        self.scaling = config.alpha / r
        self.reset_adapter()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight + self.scaling * (
            (self.B @ self.A).view_as(self.weight) + self.anchor.view_as(self.weight)
        )
        return self._conv_forward(input, weight, self.bias)

    def reset_adapter(self) -> None:
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)


class CLoRAAdapterFactory(AdapterFactory):
    def __init__(self, config: CLoRAConfig) -> None:
        super().__init__()
        self.config = config

    def _get_replacement(self, module: nn.Module) -> AdapterBase | nn.Module:
        if isinstance(module, nn.Conv2d):
            return CLoRAConv2d(
                self.config,
                module.in_channels,
                module.out_channels,
                module.kernel_size,  # type: ignore
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode,
            )
        raise ValueError(f"Unsupported layer type: {type(module)}")

    def __call__(self, module: nn.Module) -> AdapterBase | nn.Module:
        """Create an adapter for a given module."""
        replacement = self._get_replacement(module)
        if isinstance(replacement, nn.Module):
            replacement.load_state_dict(module.state_dict(), strict=False)
        return replacement


@torch.no_grad()
def update_anchors(module: nn.Module) -> None:
    """Update the anchor parameters with the current LoRA parameters."""
    for name, submodule in iter_named_adapters(module):
        if isinstance(submodule, CLoRAModule):
            submodule.anchor += submodule.B.detach() @ submodule.A.detach()
            submodule.reset_adapter()


def clora_loss(module: nn.Module) -> Tensor:
    loss = sum(
        m.self_regularization_loss()
        for _, m in iter_named_adapters(module)
        if isinstance(m, CLoRAModule)
    )
    assert isinstance(loss, Tensor)
    return loss
