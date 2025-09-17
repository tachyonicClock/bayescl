from torch import nn
from typeguard import typechecked

from bayescl.peft._base import AdapterBase, AdapterFactory
from bayescl.peft._blob import layer
from bayescl.peft._blob.config import BALLConfig
from bayescl.vbnn import VBNNConfig as VBNNConfig


class BALL(AdapterFactory):
    """A factory for creating BALL adapters."""

    @typechecked
    def __init__(self, config: BALLConfig = BALLConfig()):
        self.config = config

    @staticmethod
    def from_kwargs(**kwargs):
        return BALL(BALLConfig(**kwargs))

    def _get_replacement(self, module: nn.Module) -> AdapterBase | nn.Module:
        if isinstance(module, nn.Linear):
            return layer.BALLLinear(
                module.in_features, module.out_features, config=self.config
            )
        else:
            raise ValueError(f"Unsupported layer type: {type(module)}")

    def __call__(self, module: nn.Module) -> AdapterBase | nn.Module:
        """Create an adapter for a given module."""
        replacement = self._get_replacement(module)
        if isinstance(replacement, nn.Module):
            replacement.load_state_dict(module.state_dict(), strict=False)
        return replacement
