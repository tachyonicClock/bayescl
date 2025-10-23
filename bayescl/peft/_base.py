from typing import Protocol

from torch import nn


class AdapterBase:
    adapter_parameter_names: tuple[str, ...] = ()
    """Parameter names that are considered adapter parameters.

    This is used to identify parameters that should be treated as adapter
    parameters, which have different behavior (e.g., requiring gradients).
    """


class AdapterFactory(Protocol):
    """A callable that creates an adapter for a given module."""

    def __call__(self, module: nn.Module) -> nn.Module:
        """Create an adapter for a given module copying its state."""
        ...
