from torch import nn
from typeguard import typechecked

from bayescl.peft._ball import layer
from bayescl.peft._base import AdapterFactory
from bayescl.vbnn import VBNNConfig as VBNNConfig

from .config import BALLConfig


class BALL(AdapterFactory):
    """A factory for creating BALL adapters."""

    @typechecked
    def __init__(self, config: BALLConfig) -> None:
        self.config = config

    def _get_replacement(self, module: nn.Module) -> nn.Module:
        if isinstance(module, nn.Linear):
            return layer.BALLLinear(
                module.in_features, module.out_features, config=self.config
            )
        elif isinstance(module, nn.Conv2d):
            return layer.BALLConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                config=self.config,
            )
        else:
            raise ValueError(f"Unsupported layer type: {type(module)}")

    def __call__(self, module: nn.Module) -> nn.Module:
        """Create an adapter for a given module."""
        replacement = self._get_replacement(module)
        if isinstance(replacement, nn.Module):
            replacement.load_state_dict(module.state_dict(), strict=False)
        return replacement
