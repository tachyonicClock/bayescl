import math

import torch
from torch import nn
from torch.nn.modules.utils import _pair

from bayescl.peft._base import AdapterBase, AdapterFactory

from ._config import SDLoRAConfig


class SDLoRAModule(AdapterBase):
    adapter_parameters = ("M", "A_t", "B_t")


class SDLoRALinear(nn.Linear, SDLoRAModule):
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


class SDLoRAConv2d(nn.Conv2d, SDLoRAModule):
    AB: torch.Tensor  # Cache for frozen directions of previous tasks

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        rank_per_task: int,
        n_tasks: int,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,  # type: ignore[arg-type]
            **kwargs,
        )
        kh, kw = _pair(kernel_size)
        r = rank_per_task
        in_c_per_group = in_channels // self.groups

        # Current task trainable direction (A_t and B_t)
        self.A_t = nn.Parameter(torch.empty((r, in_c_per_group * kh * kw)))
        self.B_t = nn.Parameter(torch.empty((out_channels, r)))

        # Cache for frozen directions of previous tasks
        # Use register_buffer instead of Parameter(requires_grad=False)
        self.register_buffer("AB", torch.zeros(n_tasks, *self.weight.shape))

        # Magnitude parameters for each task
        # Page 5: "...the learned magnitudes, all initialized to ones..."
        self.M = nn.Parameter(torch.ones((n_tasks, 1)))

        self.task_idx = 0
        self._init_lora_params()

    def _init_lora_params(self):
        nn.init.normal_(self.A_t)
        nn.init.normal_(self.B_t)

    @torch.no_grad()
    def set_task(self, task: int):
        # Capture the finished direction from the task just completed
        W_finished = self.B_t @ self.A_t
        direction = W_finished / (torch.norm(W_finished, p="fro") + 1e-8)
        self.AB[self.task_idx] = direction.view_as(self.weight)

        # Reset A and B for the new task
        self._init_lora_params()
        self.task_idx = task

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Start with the frozen pre-trained weights
        W_eff = self.weight.clone()

        # Add past tasks' frozen directions scaled by their updated magnitudes
        if self.task_idx > 0:
            # Reshape M for broadcasting: (task_idx, 1, 1, 1, 1)
            past_magnitudes = self.M[: self.task_idx].view(-1, 1, 1, 1, 1)
            past_directions = self.AB[: self.task_idx]

            # Sum across the task dimension (dim=0)
            W_eff = W_eff + (past_magnitudes * past_directions).sum(dim=0)

        # Add the current task's normalized direction scaled by its magnitude
        W_t = self.B_t @ self.A_t
        W_t_normalized = W_t / (torch.norm(W_t, p="fro") + 1e-8)

        # Ensure broadcasting matches the weight dimensions
        current_magnitude = self.M[self.task_idx].view(1, 1, 1, 1)
        W_eff = W_eff + current_magnitude * W_t_normalized.view_as(self.weight)

        return nn.functional.conv2d(
            input,
            W_eff,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class SDLoRAAdapterFactory(AdapterFactory):
    """
    Wu, Y., Piao, H., Huang, L.-K., Wang, R., Li, W., Pfister, H., Meng, D., Ma, K., &
    Wei, Y. (2025). SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class
    Incremental Learning (arXiv:2501.13198). arXiv.
    https://doi.org/10.48550/arXiv.2501.13198

    Gemini 3 Summary of SDLoRA: "SD-LoRA works by splitting the learning process of a
    large AI model into two independent parts: direction and magnitude. While standard
    fine-tuning updates everything at once—often causing new information to overwrite
    and "corrupt" old memories—SD-LoRA freezes the core logic (the direction) of
    previous tasks and only allows the model to adjust how much that logic is emphasized
    (the magnitude) while it learns new patterns. By separating these components, the
    model follows a "low-loss trajectory" that discovers a shared sweet spot where new
    skills can be added without damaging the foundations of what was learned before. To
    prevent the model from becoming too bulky over time, it employs "Scalable" features
    like rank reduction for later tasks and a fusion mechanism that merges new knowledge
    into existing structures if they are mathematically similar, allowing the model to
    grow indefinitely without requiring old data for rehearsal."
    """

    def __init__(self, n_tasks: int, config: SDLoRAConfig) -> None:
        self.config = config
        self.n_tasks = n_tasks

    def _get_replacement(self, module: nn.Module) -> nn.Module:
        if isinstance(module, nn.Linear):
            return SDLoRALinear(
                module.in_features,
                module.out_features,
                rank_per_task=self.config.rank_per_task,
                n_tasks=self.n_tasks,
                bias=module.bias is not None,
            )
        elif isinstance(module, nn.Conv2d):
            return SDLoRAConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                rank_per_task=self.config.rank_per_task,
                n_tasks=self.n_tasks,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
            )
        else:
            raise ValueError(f"Unsupported layer type: {type(module)}")

    def __call__(self, module: nn.Module) -> nn.Module:
        """Create an adapter for a given module."""
        replacement = self._get_replacement(module)
        if isinstance(replacement, nn.Module):
            replacement.load_state_dict(module.state_dict(), strict=False)
        return replacement


def set_task(module: nn.Module, task: int) -> None:
    """Set the active task for all SD-LoRA modules in the given module."""
    for submodule in module.modules():
        if isinstance(submodule, (SDLoRALinear, SDLoRAConv2d)):
            submodule.set_task(task)
