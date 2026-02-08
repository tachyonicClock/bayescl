from dataclasses import dataclass

from torch import nn

from bayescl.peft._base import AdapterBase, AdapterFactory
from bayescl.peft._lora import layer


@dataclass
class LoRA_Factory(AdapterFactory):
    r: int = 16
    lora_alpha: int = 1
    lora_dropout: float = 0.0

    def _get_replacement(self, module: nn.Module) -> AdapterBase | nn.Module:
        match module:
            case nn.Linear():
                return layer.Linear(
                    module.in_features,
                    module.out_features,
                    r=self.r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    merge_weights=False,
                )
            case nn.Conv1d():
                return layer.Conv1d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    r=self.r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    merge_weights=False,
                )
            case nn.Conv2d():
                return layer.Conv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    r=self.r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    merge_weights=False,
                )
            case _:
                raise ValueError(f"Unsupported layer type: {type(module)}")

    def __call__(self, module: nn.Module) -> AdapterBase | nn.Module:
        replacement = self._get_replacement(module)
        if isinstance(replacement, nn.Module):
            replacement.load_state_dict(module.state_dict(), strict=False)
        return replacement
