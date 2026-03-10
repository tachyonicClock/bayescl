from torch import nn

from bayescl.config import SDLoRAConfig
from bayescl.peft._base import AdapterFactory

from .layer import SDLoRAConv2d, SDLoRALinear


class SDLoRA(AdapterFactory):
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
