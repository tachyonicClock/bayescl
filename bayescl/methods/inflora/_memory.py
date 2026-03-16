"""Memory management for InfLoRA using Dual GPM strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ._constants import SVD_EPSILON


@dataclass
class LayerMemoryState:
    """State of memory for a single layer."""

    basis: Tensor
    project_type: str


class DualGPMMemory:
    """Dual GPM-based memory for managing task-specific basis vectors."""

    def __init__(
        self,
        threshold_start: float = 0.90,
        threshold_end: float = 0.98,
        total_tasks: Optional[int] = None,
        eps: float = SVD_EPSILON,
    ) -> None:
        if not 0.0 < threshold_start <= 1.0:
            raise ValueError("threshold_start must be in (0, 1]")
        if not 0.0 < threshold_end <= 1.0:
            raise ValueError("threshold_end must be in (0, 1]")
        self.threshold_start = threshold_start
        self.threshold_end = threshold_end
        self.total_tasks = total_tasks
        self.eps = eps
        self.layers: dict[str, LayerMemoryState] = {}

    def threshold(self, task_index: int) -> float:
        """Compute threshold for the given task index."""
        if self.total_tasks is None or self.total_tasks <= 1:
            return self.threshold_start
        progress = min(max(task_index, 0), self.total_tasks) / self.total_tasks
        return (
            self.threshold_start
            + (self.threshold_end - self.threshold_start) * progress
        )

    def project_new_task(self, layer_name: str, activation: Tensor) -> Tensor:
        """Project new task activation to orthogonal space."""
        activation = activation.detach().to(device="cpu", dtype=torch.float32)
        state = self.layers.get(layer_name)
        if state is None or state.basis.numel() == 0:
            return activation
        projector = state.basis @ state.basis.t()
        if state.project_type == "remove":
            return activation - projector @ activation
        return projector @ activation

    def allowed_residual_basis(
        self, layer_name: str, feature_dim: int, rank: int
    ) -> Tensor:
        """Get allowed basis for residual space."""
        state = self.layers.get(layer_name)
        if state is None:
            return torch.eye(feature_dim, dtype=torch.float32)[:rank]
        if state.project_type == "retain":
            return state.basis[:, : min(rank, state.basis.shape[1])].t().contiguous()
        complement = self._orthonormal_complement(state.basis, feature_dim)
        if complement.numel() == 0:
            return torch.empty(0, feature_dim, dtype=torch.float32)
        return complement[:, : min(rank, complement.shape[1])].t().contiguous()

    def update(self, layer_name: str, activation: Tensor, task_index: int) -> None:
        """Update memory with new task activation."""
        activation = activation.detach().to(device="cpu", dtype=torch.float32)
        if activation.numel() == 0 or activation.shape[1] == 0:
            return
        threshold = self.threshold(task_index)
        state = self.layers.get(layer_name)
        if state is None:
            self.layers[layer_name] = self._initial_state(activation, threshold)
            return
        if state.project_type == "remove":
            updated_basis = self._update_remove_basis(
                state.basis, activation, threshold
            )
            if updated_basis.shape[1] > activation.shape[0] / 2:
                self.layers[layer_name] = LayerMemoryState(
                    basis=self._orthonormal_complement(
                        updated_basis, activation.shape[0]
                    ),
                    project_type="retain",
                )
            else:
                self.layers[layer_name] = LayerMemoryState(
                    basis=updated_basis,
                    project_type="remove",
                )
            return

        updated_basis = self._update_retain_basis(state.basis, activation, threshold)
        self.layers[layer_name] = LayerMemoryState(
            basis=updated_basis,
            project_type="retain",
        )

    def _initial_state(self, activation: Tensor, threshold: float) -> LayerMemoryState:
        """Compute initial memory state."""
        basis, _ = self._principal_subspace(activation, threshold)
        basis = self._orthonormalize(basis)
        if basis.shape[1] <= activation.shape[0] / 2:
            return LayerMemoryState(basis=basis, project_type="remove")
        return LayerMemoryState(
            basis=self._orthonormal_complement(basis, activation.shape[0]),
            project_type="retain",
        )

    def _update_remove_basis(
        self, basis: Tensor, activation: Tensor, threshold: float
    ) -> Tensor:
        """Update basis when in remove mode."""
        total_energy = self._energy(activation)
        residual = activation - (basis @ basis.t()) @ activation
        candidate_basis, singular_values = self._principal_subspace(residual, None)
        if singular_values.numel() == 0 or total_energy <= self.eps:
            return basis
        energy_ratio = (singular_values.square() / total_energy).tolist()
        covered = (total_energy - self._energy(residual)) / total_energy
        rank = 0
        while rank < len(energy_ratio) and covered < threshold:
            covered += energy_ratio[rank]
            rank += 1
        if rank == 0:
            return basis
        expanded = torch.cat([basis, candidate_basis[:, :rank]], dim=1)
        return self._orthonormalize(expanded)

    def _update_retain_basis(
        self, basis: Tensor, activation: Tensor, threshold: float
    ) -> Tensor:
        """Update basis when in retain mode."""
        total_energy = self._energy(activation)
        retained = (basis @ basis.t()) @ activation
        candidate_basis, singular_values = self._principal_subspace(retained, None)
        if singular_values.numel() == 0 or total_energy <= self.eps:
            return basis
        energy_ratio = (singular_values.square() / total_energy).tolist()
        retained_ratio = self._energy(retained) / total_energy
        rank = 0
        while rank < len(energy_ratio) and retained_ratio >= (1 - threshold):
            retained_ratio -= energy_ratio[rank]
            rank += 1
        if rank == 0:
            return basis
        reduced = (
            basis - (candidate_basis[:, :rank] @ candidate_basis[:, :rank].t()) @ basis
        )
        return self._orthonormalize(reduced)

    def _principal_subspace(
        self, activation: Tensor, threshold: Optional[float]
    ) -> tuple[Tensor, Tensor]:
        """Compute principal subspace of activation."""
        if activation.numel() == 0:
            return (
                torch.empty(activation.shape[0], 0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
            )
        u, s, _ = torch.linalg.svd(activation, full_matrices=False)
        keep = s > self.eps
        if keep.sum() == 0:
            return (
                torch.empty(activation.shape[0], 0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
            )
        u = u[:, keep]
        s = s[keep]
        if threshold is None:
            return u, s
        ratio = s.square() / s.square().sum()
        rank = int(torch.sum(torch.cumsum(ratio, dim=0) < threshold).item())
        rank = max(rank, 1)
        return u[:, :rank], s[:rank]

    def _orthonormalize(self, basis: Tensor) -> Tensor:
        """Orthonormalize basis using SVD."""
        if basis.numel() == 0:
            return torch.empty(basis.shape[0], 0, dtype=torch.float32)
        u, s, _ = torch.linalg.svd(basis, full_matrices=False)
        keep = s > self.eps
        if keep.sum() == 0:
            return torch.empty(basis.shape[0], 0, dtype=torch.float32)
        return u[:, keep]

    def _orthonormal_complement(self, basis: Tensor, feature_dim: int) -> Tensor:
        """Compute orthonormal complement of basis."""
        if basis.numel() == 0:
            return torch.eye(feature_dim, dtype=torch.float32)
        q, _ = torch.linalg.qr(basis, mode="complete")
        return q[:, basis.shape[1] :]

    def _energy(self, activation: Tensor) -> float:
        """Compute energy (Frobenius norm squared) of activation."""
        return float(torch.sum(torch.square(activation)).item())
