"""Base class for InfLoRA adapters."""

from __future__ import annotations

from typing import Iterator, cast

import torch
from torch import Tensor, nn

from bayescl.peft import AdapterBase

from ._constants import DEFAULT_MAX_COLLECTED_SAMPLES


class _LowRankBranch(nn.Module):
    """A low-rank adapter branch for a single task."""

    basis: Tensor
    adapter: nn.Parameter

    def __init__(self, basis: Tensor, out_features: int) -> None:
        super().__init__()
        if basis.ndim != 2:
            raise ValueError("basis must have shape [rank, in_features]")
        self.register_buffer("basis", basis.detach().clone())
        self.adapter = nn.Parameter(
            torch.zeros(
                out_features,
                basis.shape[0],
                dtype=basis.dtype,
                device=basis.device,
            )
        )

    def delta_weight(self) -> Tensor:
        return self.adapter @ self.basis


class InfLoRAAdapterBase(AdapterBase):
    """Base class for InfLoRA adapters (Linear and Conv2d)."""

    adapter_parameters = ()

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.branches = nn.ModuleDict()
        self.active_task: str | None = None
        self.collect_inputs = False
        self._input_snapshots: list[Tensor] = []
        self._max_collected_samples = DEFAULT_MAX_COLLECTED_SAMPLES

    def enable_input_collection(self) -> None:
        self.collect_inputs = True

    def disable_input_collection(self) -> None:
        self.collect_inputs = False

    def clear_inputs(self) -> None:
        self._input_snapshots.clear()

    def consume_inputs(self, max_samples: int | None = None) -> Tensor:
        max_samples = max_samples or self._max_collected_samples
        if not self._input_snapshots:
            return torch.empty(self.in_features, 0, dtype=self.weight.dtype)
        inputs = torch.cat(self._input_snapshots, dim=0)
        if inputs.shape[0] > max_samples:
            inputs = inputs[:max_samples]
        self._input_snapshots.clear()
        return inputs.t().contiguous()

    def set_active_task(self, task_id: str | int | None) -> None:
        self.active_task = None if task_id is None else str(task_id)
        for name, branch in self.branches.items():
            branch = cast(_LowRankBranch, branch)
            branch.adapter.requires_grad_(name == self.active_task)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        if self.active_task is None:
            return
        branch = cast(_LowRankBranch, self.branches[self.active_task])
        yield branch.adapter

    def merge_task(self, task_id: str | int | None = None, unload: bool = True) -> None:
        task_key = self.active_task if task_id is None else str(task_id)
        if task_key is None:
            return
        branch = cast(_LowRankBranch, self.branches[task_key])
        with torch.no_grad():
            self.weight.add_(
                self._branch_delta_to_weight(branch).to(
                    dtype=self.weight.dtype,
                    device=self.weight.device,
                )
            )
        if unload:
            del self.branches[task_key]
        if self.active_task == task_key:
            self.active_task = None

    def _branch_delta_to_weight(self, branch: _LowRankBranch) -> Tensor:
        """Convert branch delta weight to weight shape."""
        return branch.delta_weight()

    def _snapshot_inputs(self, inputs: Tensor) -> None:
        """Snapshot inputs for later use. To be overridden by subclasses."""
        raise NotImplementedError

    def _add_adapter_output(self, outputs: Tensor, inputs: Tensor) -> Tensor:
        """Add adapter outputs to main outputs. To be overridden by subclasses."""
        raise NotImplementedError
