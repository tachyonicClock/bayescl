import math

import torch
from torch import Tensor, nn

from bayescl.peft import AdapterBase, AdapterFactory, iter_named_adapters

from ._config import CLoRAConfig


class CLoRAModule(AdapterBase):
    """Shared c-LoRA logic for a single adapted weight matrix.

    The adapted weight is treated as a 2D matrix of shape ``(fan_out, fan_in)``
    (``nn.Conv2d`` kernels are flattened to ``(out_channels, in_channels // groups
    * kh * kw)``).  The task-``t`` update is the genuine rank-``r`` product
    :math:`\\mathbf{B}_t \\mathbf{A}_t` with :math:`\\mathbf{A}_t \\in
    \\mathbb{R}^{r \\times \\text{fan\\_in}}` and :math:`\\mathbf{B}_t \\in
    \\mathbb{R}^{\\text{fan\\_out} \\times r}`, matching Eq. 2 of the paper.
    """

    adapter_parameters = ("A", "B")
    adapter_buffers = ("anchor_A", "anchor_B")

    n_tasks_seen: int = 0
    n_tasks: int = 0
    A: nn.Parameter
    B: nn.Parameter
    anchor_A: Tensor
    anchor_B: Tensor

    def _init_clora(
        self, config: CLoRAConfig, fan_in: int, fan_out: int, n_tasks: int
    ) -> None:
        r = config.rank
        self.A = nn.Parameter(torch.empty(r, fan_in))
        self.B = nn.Parameter(torch.empty(fan_out, r))
        self.anchor_A = nn.Buffer(torch.zeros(n_tasks, r, fan_in))
        self.anchor_B = nn.Buffer(torch.zeros(n_tasks, fan_out, r))
        self.n_tasks = n_tasks
        self.n_tasks_seen = 0
        self.scaling = config.alpha / r
        self.reset_adapter()

    def reset_adapter(self) -> None:
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def delta(self) -> Tensor:
        """Current task update :math:`\\mathbf{B}_t \\mathbf{A}_t`."""
        return self.B @ self.A

    def get_anchor_product(self) -> Tensor:
        r"""Sum of all accumulated anchor products :math:`\sum_i \mathbf{B}_i \mathbf{A}_i`."""
        limit = min(self.n_tasks_seen, self.n_tasks)
        if limit > 0:
            return torch.einsum(
                "tij,tjk->ik", self.anchor_B[:limit], self.anchor_A[:limit]
            )
        return torch.zeros(
            self.B.shape[0], self.A.shape[1], device=self.A.device, dtype=self.A.dtype
        )

    def adapted_weight(self, weight: Tensor) -> Tensor:
        """Apply Eq. 2: ``W_init + scaling * (accumulated deltas + current delta)``."""
        update = self.scaling * (self.delta() + self.get_anchor_product())
        return weight + update.view_as(weight)

    def self_regularization_loss(self) -> Tensor:
        r"""Penalize changes to features learned by previous tasks (Eq. 3).

        .. math::

            \mathcal{L}_{\text{reg}} =
            || \left| \sum_{t'=1}^{t-1} \mathbf{B}_{t'} \mathbf{A}_{t'} \right|
            \odot \mathbf{B}_t \mathbf{A}_t ||_F^2

        The scaling factor :math:`\alpha / r` is deliberately omitted here; it is a
        constant that is absorbed into the ``lambda_`` penalty weight.
        """
        anchor_product = self.get_anchor_product()
        return (anchor_product.abs() * self.delta()).norm(p="fro") ** 2


class CLoRALinear(nn.Linear, CLoRAModule):
    def __init__(
        self,
        config: CLoRAConfig,
        in_features: int,
        out_features: int,
        n_tasks: int,
        **kwargs,
    ) -> None:
        super().__init__(in_features, out_features, **kwargs)
        self._init_clora(config, in_features, out_features, n_tasks)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(input, self.adapted_weight(self.weight), self.bias)


class CLoRAConv2d(nn.Conv2d, CLoRAModule):
    def __init__(
        self,
        config: CLoRAConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        n_tasks: int,
        **kwargs,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        kh, kw = self.kernel_size
        fan_in = (in_channels // self.groups) * kh * kw
        self._init_clora(config, fan_in, out_channels, n_tasks)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(
            input, self.adapted_weight(self.weight), self.bias
        )


class CLoRAAdapterFactory(AdapterFactory):
    def __init__(self, n_tasks: int, config: CLoRAConfig) -> None:
        super().__init__()
        self.config = config
        self.n_tasks = n_tasks

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
                n_tasks=self.n_tasks,
            )
        if isinstance(module, nn.Linear):
            return CLoRALinear(
                self.config,
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                n_tasks=self.n_tasks,
            )
        raise ValueError(f"Unsupported layer type: {type(module)}")

    def __call__(self, module: nn.Module) -> AdapterBase | nn.Module:
        """Create an adapter for a given module."""
        replacement = self._get_replacement(module)
        if isinstance(replacement, nn.Module):
            replacement.load_state_dict(module.state_dict(), strict=False)
        return replacement


@torch.no_grad()
def update_anchors(module: nn.Module, task_index: int) -> None:
    """Update the anchor parameters with the current LoRA parameters.

    Stores the current A and B parameters at the specified task index.
    """
    for name, submodule in iter_named_adapters(module):
        if isinstance(submodule, CLoRAModule):
            submodule.anchor_A[task_index] = submodule.A.detach()
            submodule.anchor_B[task_index] = submodule.B.detach()
            submodule.n_tasks_seen = max(submodule.n_tasks_seen, task_index + 1)
            submodule.reset_adapter()


def clora_loss(module: nn.Module) -> Tensor:
    loss: Tensor = torch.zeros(())
    for _, m in iter_named_adapters(module):
        if isinstance(m, CLoRAModule):
            reg = m.self_regularization_loss()
            loss = loss.to(reg) + reg
    return loss
