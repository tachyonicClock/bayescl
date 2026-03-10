"""Module to create and manage PEFT adapters."""

import re
from typing import Any, Generator, Protocol, Tuple

from torch import nn

from bayescl.peft._base import AdapterBase, AdapterFactory


def set_module(module: nn.Module, name: str, new_module: Any) -> None:
    """A helper function to set a submodule of a module by its name.

    The name should be a string with the format `parent.child`
    (`module.submodule.child`) where "parent" is the given module and "child" is
    the submodule to be replaced.

    :param module: The parent module.
    :param name: A submodule name in the format "parent.child".
    :param new_module: The new module to replace the old one.
    """
    if name.count(".") == 0:
        setattr(module, name, new_module)
    else:
        parent, child = name.rsplit(".", 1)
        parent_module = module.get_submodule(parent)
        setattr(parent_module, child, new_module)


class AdapterFilter(Protocol):
    def __call__(self, name: str, module: nn.Module) -> bool:
        """Should the module be replaced with an adapter?"""
        ...


class RegexFilter(AdapterFilter):
    def __init__(self, pattern: str) -> None:
        self.pattern = re.compile(pattern)

    def __call__(self, name: str, module: nn.Module) -> bool:
        return self.pattern.match(name) is not None


def add_adapters[T: nn.Module](
    module: T, filter: AdapterFilter, factory: AdapterFactory
) -> T:
    """Create and replace submodules with adapters.

    :param module: The module to modify in-place.
    :param filter: A callable that filters submodules to apply the adapter.
    :param factory: A callable that creates an adapter module for the given
        submodule.
    """
    for name, submodule in module.named_modules():
        if filter(name, submodule):
            set_module(module, name, factory(submodule))
    only_adapters_require_grad(module)
    return module


def iter_named_adapters(
    module: nn.Module,
) -> Generator[Tuple[str, AdapterBase], None, None]:
    """Iterate over all adapter modules (:class:`AdapterBase`)"""
    for name, submodule in module.named_modules():
        if isinstance(submodule, AdapterBase):
            yield name, submodule


def iter_adapter_parameters(
    module: nn.Module,
) -> Generator[Tuple[str, nn.Parameter], None, None]:
    """Iterate over all parameters in adapters.

    See :attr:`AdapterBase.adapter_parameter_names` for what is considered an
    adapter parameter.
    """
    for prefix, submodule in iter_named_adapters(module):
        if isinstance(submodule, nn.Module):
            for name in submodule.adapter_parameter_names:
                yield f"{prefix}.{name}", submodule.get_parameter(name)
            for name in submodule.adapter_modules:
                child_module = submodule.get_submodule(name)
                if isinstance(child_module, nn.Module):
                    for param_name, param in child_module.named_parameters():
                        yield f"{prefix}.{name}.{param_name}", param


def only_adapters_require_grad(module: nn.Module) -> None:
    """Disable grad unless parameters are in adapters.

    See :attr:`AdapterBase.adapter_parameter_names` for what is considered an
    adapter parameter.
    """
    module.requires_grad_(False)
    for _, param in iter_adapter_parameters(module):
        param.requires_grad_(True)


def count_adapter_parameters(module: nn.Module) -> int:
    """Count the number of parameters in adapters.

    See :attr:`AdapterBase.adapter_parameter_names` for what is considered an
    adapter parameter.
    """
    return sum(p.numel() for _, p in iter_adapter_parameters(module))


def parameter_summary_str(module: nn.Module) -> str:
    """Print a summary of the number of parameters in the module."""
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    buffers = sum(p.numel() for p in module.buffers())
    frozen_params = total_params - trainable_params
    adapter_params = count_adapter_parameters(module)

    fmt_int = ">10,"
    fmt_percent = ">9.2f"
    return (
        f"Total:       {total_params:{fmt_int}}\n"
        f"Trainable:   {trainable_params:{fmt_int}}\n"
        f"Frozen:      {frozen_params:{fmt_int}}\n"
        f"Buffers:     {buffers:{fmt_int}}\n"
        f"Adapter:     {adapter_params:{fmt_int}}\n"
        f"Non-adapter: {(total_params - adapter_params):{fmt_int}}\n"
        f"Adapter %:   {(adapter_params / total_params * 100):{fmt_percent}}%\n"
        f"Trainable %: {(trainable_params / total_params * 100):{fmt_percent}}%\n"
    )
