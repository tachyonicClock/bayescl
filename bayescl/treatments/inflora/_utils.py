"""Utility functions for InfLoRA."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from ._constants import SVD_EPSILON


def iter_inflora_layers(module: nn.Module) -> Iterable[tuple[str, object]]:
    """Iterate over all InfLoRA adapter layers in a module.

    Yields:
        Tuples of (layer_name, layer) for all InfLoRA layers.
    """
    # Import here to avoid circular dependency
    from ._module import InfLoRAConv2d, InfLoRALinear

    for name, child in module.named_modules():
        if isinstance(child, (InfLoRALinear, InfLoRAConv2d)):
            yield name, child


def extract_model_inputs(batch: object) -> Tensor:
    """Extract input tensor from various batch formats.

    Args:
        batch: Single tensor, dict with tensors, or sequence with tensors.

    Returns:
        The extracted input tensor.

    Raises:
        TypeError: If no suitable tensor found in batch.
    """
    if isinstance(batch, Tensor):
        return batch
    if isinstance(batch, dict):
        for value in batch.values():
            if isinstance(value, Tensor):
                return value
        raise TypeError("no tensor input found in batch dictionary")
    if isinstance(batch, (list, tuple)):
        tensors = [item for item in batch if isinstance(item, Tensor)]
        if not tensors:
            raise TypeError("no tensor input found in batch sequence")
        for tensor in tensors:
            if tensor.ndim >= 2 and tensor.is_floating_point():
                return tensor
        for tensor in tensors:
            if tensor.ndim >= 2:
                return tensor
        return tensors[0]
    raise TypeError(f"unsupported batch type: {type(batch)!r}")


def collect_task_activations(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Tensor]:
    """Collect layer activations from data loader.

    Args:
        model: Neural network model with InfLoRA layers.
        data_loader: DataLoader for collecting activations.
        device: Device to run the model on.
        max_batches: Maximum number of batches to process. None for all.

    Returns:
        Dictionary mapping layer names to activation tensors.
    """
    layers = list(iter_inflora_layers(model))
    for _, layer in layers:
        layer.clear_inputs()
        layer.enable_input_collection()

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            inputs = extract_model_inputs(batch).to(device)
            model(inputs)
            if max_batches is not None and batch_index + 1 >= max_batches:
                break

    activations = {name: layer.consume_inputs() for name, layer in layers}
    for _, layer in layers:
        layer.disable_input_collection()
    model.train(was_training)
    return activations


def extract_principal_basis(activation: Tensor, max_rank: int) -> Tensor:
    """Extract principal basis rows from activation matrix.

    Args:
        activation: Activation matrix to decompose.
        max_rank: Maximum rank to extract.

    Returns:
        Principal basis vectors as rows.
    """
    if activation.numel() == 0 or activation.shape[1] == 0:
        return torch.empty(0, activation.shape[0], dtype=torch.float32)
    u, s, _ = torch.linalg.svd(activation, full_matrices=False)
    keep = int((s > SVD_EPSILON).sum().item())
    if keep == 0:
        return torch.empty(0, activation.shape[0], dtype=torch.float32)
    rank = min(max_rank, keep)
    return u[:, :rank].t().contiguous()
