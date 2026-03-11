from abc import ABC, abstractmethod

from torch import nn


class AdapterBase:
    adapter_parameters: tuple[str, ...] = ()
    """Names of tunable parameters in the adapter."""
    adapter_buffers: tuple[str, ...] = ()
    """Names of tunable buffers in the adapter."""


class AdapterFactory(ABC):
    """A callable that creates an adapter for a given module."""

    @abstractmethod
    def __call__(self, module: nn.Module) -> AdapterBase | nn.Module:
        """Create an adapter for a given module copying its state."""
        ...
