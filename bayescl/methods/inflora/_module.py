"""InfLoRA layer implementations and engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bayescl.peft import AdapterBase, AdapterFactory

from ._adapter_base import InfLoRAAdapterBase, _LowRankBranch
from ._config import InfLoRAConfig
from ._memory import DualGPMMemory
from ._utils import collect_task_activations, extract_principal_basis


class InfLoRALinear(nn.Linear, InfLoRAAdapterBase):
    """Linear layer with InfLoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        nn.Linear.__init__(self, in_features, out_features, bias, device, dtype)
        InfLoRAAdapterBase.__init__(self, rank)

    def expand_task(self, task_id: str | int, basis: Tensor) -> None:
        """Add a new task-specific adapter branch."""
        task_key = str(task_id)
        if task_key in self.branches:
            raise ValueError(f"task {task_key} already exists")
        if basis.ndim != 2 or basis.shape[1] != self.in_features:
            raise ValueError("basis must have shape [rank, in_features]")
        basis = basis.to(device=self.weight.device, dtype=self.weight.dtype)
        self.branches[task_key] = _LowRankBranch(basis, self.out_features)
        self.set_active_task(task_key)

    def _snapshot_inputs(self, inputs: Tensor) -> None:
        """Collect input snapshots for basis computation."""
        flattened = (
            inputs.reshape(-1, inputs.shape[-1])
            .detach()
            .to(device="cpu", dtype=torch.float32)
        )
        current = sum(chunk.shape[0] for chunk in self._input_snapshots)
        if current >= self._max_collected_samples:
            return
        remaining = self._max_collected_samples - current
        if flattened.shape[0] > remaining:
            flattened = flattened[:remaining]
        self._input_snapshots.append(flattened)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass with adapter outputs."""
        if self.collect_inputs:
            self._snapshot_inputs(inputs)
        outputs = F.linear(inputs, self.weight, self.bias)
        for branch in self.branches.values():
            outputs = outputs + branch(inputs)
        return outputs


class InfLoRAConv2d(nn.Conv2d, InfLoRAAdapterBase):
    """Conv2d layer with InfLoRA adapters."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        rank: int,
        **kwargs,
    ) -> None:
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        InfLoRAAdapterBase.__init__(self, rank)

    @property
    def in_features(self) -> int:
        """Compute in_features from conv2d parameters."""
        kh, kw = self.kernel_size
        return (self.in_channels // self.groups) * kh * kw

    def expand_task(self, task_id: str | int, basis: Tensor) -> None:
        """Add a new task-specific adapter branch."""
        task_key = str(task_id)
        if task_key in self.branches:
            raise ValueError(f"task {task_key} already exists")
        if basis.ndim != 2 or basis.shape[1] != self.in_features:
            raise ValueError("basis must have shape [rank, in_features]")
        basis = basis.to(device=self.weight.device, dtype=self.weight.dtype)
        self.branches[task_key] = _LowRankBranch(basis, self.out_channels)
        self.set_active_task(task_key)

    def _snapshot_inputs(self, inputs: Tensor) -> None:
        """Collect input patches for basis computation."""
        patches = F.unfold(
            inputs.detach(),
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=cast(tuple[int, int], self.padding),
            stride=self.stride,
        )
        flattened = (
            patches.transpose(1, 2)
            .reshape(-1, patches.shape[1])
            .to(device="cpu", dtype=torch.float32)
        )
        current = sum(chunk.shape[0] for chunk in self._input_snapshots)
        if current >= self._max_collected_samples:
            return
        remaining = self._max_collected_samples - current
        if flattened.shape[0] > remaining:
            flattened = flattened[:remaining]
        self._input_snapshots.append(flattened)

    def _branch_delta_to_weight(self, branch: _LowRankBranch) -> Tensor:
        """Convert branch delta to weight shape for conv2d."""
        return branch.delta_weight().view_as(self.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass with adapter outputs."""
        if self.collect_inputs:
            self._snapshot_inputs(inputs)
        weight = self.weight
        for branch in self.branches.values():
            weight = weight + branch.delta_weight().view_as(self.weight)
        return self._conv_forward(inputs, weight, self.bias)


def iter_inflora_layers(
    module: nn.Module,
) -> Iterable[tuple[str, InfLoRALinear | InfLoRAConv2d]]:
    """Iterate over all InfLoRA layers in a module."""
    for name, child in module.named_modules():
        if isinstance(child, (InfLoRALinear, InfLoRAConv2d)):
            yield name, child


@dataclass
class TaskPlan:
    """Plan for a single training task."""

    task_id: str
    task_index: int
    bases: dict[str, Tensor]
    activations: dict[str, Tensor]


class InfLoRAEngine:
    """Engine for managing InfLoRA task expansion and finalization."""

    def __init__(
        self,
        model: nn.Module,
        rank: int,
        threshold_start: float = 0.90,
        threshold_end: float = 0.98,
        total_tasks: Optional[int] = None,
    ) -> None:
        self.model = model
        self.rank = rank
        self.memory = DualGPMMemory(
            threshold_start=threshold_start,
            threshold_end=threshold_end,
            total_tasks=total_tasks,
        )

    def plan_task(
        self,
        task_id: str | int,
        task_index: int,
        data_loader: Iterable,
        device: torch.device,
        max_batches: Optional[int] = None,
    ) -> TaskPlan:
        """Plan task-specific bases without modifying the model."""
        activations = collect_task_activations(
            self.model,
            data_loader,
            device=device,
            max_batches=max_batches,
        )
        bases: dict[str, Tensor] = {}
        for name, layer in iter_inflora_layers(self.model):
            projected = self.memory.project_new_task(name, activations[name])
            basis = extract_principal_basis(
                projected,
                max_rank=min(self.rank, layer.rank),
            )
            if basis.shape[0] == 0:
                basis = self.memory.allowed_residual_basis(
                    name,
                    layer.in_features,
                    rank=min(self.rank, layer.rank),
                )
            if basis.shape[0] == 0:
                basis = torch.eye(layer.in_features, dtype=torch.float32)[:1]
            bases[name] = basis

        return TaskPlan(
            task_id=str(task_id),
            task_index=task_index,
            bases=bases,
            activations=activations,
        )

    def begin_task(
        self,
        task_id: str | int,
        task_index: int,
        data_loader: Iterable,
        device: torch.device,
        max_batches: Optional[int] = None,
    ) -> TaskPlan:
        """Compute and apply task-specific bases to the model."""
        plan = self.plan_task(
            task_id=task_id,
            task_index=task_index,
            data_loader=data_loader,
            device=device,
            max_batches=max_batches,
        )
        for name, layer in iter_inflora_layers(self.model):
            layer.expand_task(
                plan.task_id,
                plan.bases[name].to(
                    dtype=layer.weight.dtype,
                    device=layer.weight.device,
                ),
            )
        return plan

    def finalize_task(
        self,
        task_id: str | int,
        task_index: int,
        data_loader: Iterable,
        device: torch.device,
        max_batches: Optional[int] = None,
    ) -> None:
        """Merge task adapters and update memory."""
        activations = collect_task_activations(
            self.model,
            data_loader,
            device=device,
            max_batches=max_batches,
        )
        for name, activation in activations.items():
            self.memory.update(name, activation, task_index=task_index)
        for _, layer in iter_inflora_layers(self.model):
            layer.merge_task(task_id)

    def parameter_count(self) -> int:
        x = 0
        for _, layer in iter_inflora_layers(self.model):
            for branch in layer.branches.values():
                assert isinstance(branch, _LowRankBranch)
                x += branch.adapter.numel() # type: ignore
        return x
    
    def buffer_count(self) -> int:
        x = 0
        for _, layer in iter_inflora_layers(self.model):
            for branch in layer.branches.values():
                assert isinstance(branch, _LowRankBranch)
                x += branch.basis.numel()  # type: ignore
        x += self.memory.buffer_count()
        return x


    def trainable_parameters(
        self,
        extra_parameters: Optional[Iterable[nn.Parameter]] = None,
    ) -> list[nn.Parameter]:
        """Get all trainable parameters including extra ones."""
        parameters: list[nn.Parameter] = []
        seen: set[int] = set()

        def append_unique(parameter: nn.Parameter) -> None:
            key = id(parameter)
            if key in seen:
                return
            seen.add(key)
            parameters.append(parameter)

        for _, layer in iter_inflora_layers(self.model):
            for parameter in layer.trainable_parameters():
                append_unique(parameter)
        if extra_parameters is not None:
            for parameter in extra_parameters:
                if parameter.requires_grad:
                    append_unique(parameter)
        return parameters


class InfLoRAAdapterFactory(AdapterFactory):
    """Factory for creating InfLoRA adapter layers."""

    def __init__(self, config: InfLoRAConfig) -> None:
        self.config = config

    def _get_replacement(self, module: nn.Module) -> AdapterBase | nn.Module:
        if isinstance(module, nn.Linear):
            return InfLoRALinear(
                module.in_features,
                module.out_features,
                rank=self.config.rank,
                bias=module.bias is not None,
            )
        if isinstance(module, nn.Conv2d):
            return InfLoRAConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,  # type: ignore[arg-type]
                rank=self.config.rank,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode,
            )
        raise ValueError(f"Unsupported layer type: {type(module)}")

    def __call__(self, module: nn.Module) -> AdapterBase | nn.Module:
        replacement = self._get_replacement(module)
        if isinstance(replacement, nn.Module):
            replacement.load_state_dict(module.state_dict(), strict=False)
        return replacement
