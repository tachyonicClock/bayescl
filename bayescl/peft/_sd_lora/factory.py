from torch import nn

from bayescl.peft._base import AdapterFactory

from .config import SDLoRAConfig
from .layer import SDLoRAConv2d, SDLoRALinear


class SDLoRA(AdapterFactory):
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

    @staticmethod
    def set_task(module: nn.Module, task: int) -> None:
        """Set the active task for all SDLoRA modules in the given module."""
        for submodule in module.modules():
            if isinstance(submodule, (SDLoRALinear, SDLoRAConv2d)):
                submodule.set_task(task)
