"""


Psudo-python describing the usage of this module:

..  code-block:: python

    gpm = None
    for t in tasks:

        # Runs before task with current task data
        projection = orthogonal_projection_matrix(sample_module_inputs(...), gpm, ...)

            # Set model's projections here ...

        # Train on task current task here ...

        # Runs after task with current task data
        module_inputs = sample_module_inputs(...)
        if t == 0:
            # Create GPM for the first task
            gpm = create_gpm(module_inputs, threshold)
        else:
            # Update GPM for subsequent tasks
            gpm = update_gpm(gpm, module_inputs, threshold)



.. [saha20] Saha, G., & Roy, K. (2020, September). Gradient Projection Memory for Continual
    Learning. International Conference on Learning Representations.
.. [liang24] Liang, Y.-S., & Li, W.-J. (2024). InfLoRA: Interference-Free Low-Rank
    Adaptation for Continual Learning. 23638–23647.
.. [Liang23] Liang, Y.-S., & Li, W.-J. (2023). Adaptive plasticity improvement for
    continual learning. IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, 7816–7825.
    https://doi.org/10.1109/CVPR52729.2023.00755

"""

import enum
import math
from functools import partial
from typing import Any, ContextManager, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


class _ProjectType(enum.Enum):
    REMOVE = 0
    RETAIN = 1


def _get_input_hat(inputs: Tensor, gpm: Tensor) -> Tensor:
    r"""Implements Equation 8 [saha20]_.

    Eliminates the common directions (bases) that are already present in the GPM so that
    newly added bases are unique and orthogonal to the existing bases in the memory.

    :param inputs: :math:`\mathbf{H}` of shape (n_samples, in_features)
    :param gpm: :math:`\mathbf{M}` of shape (in_features, k)
    """
    return inputs - inputs @ gpm @ gpm.T


class CaptureModuleInputs(ContextManager["CaptureModuleInputs"]):
    def __init__(
        self, module: nn.Module, module_names: Sequence[str], device: torch.device
    ) -> None:
        super().__init__()
        self._module = module
        self._module_names = module_names
        self._device = device
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._module_inputs: Dict[str, Tuple[Tensor, int]] = {}

    def _record(
        self, module: nn.Module, args: Tuple[Tensor, ...], output: Tensor, *, name: str
    ) -> None:
        x = args[0]
        cur_matrix, n_cur_matrix = self._module_inputs[name]

        if x.ndim == 3:
            cur_matrix = (
                cur_matrix * n_cur_matrix + torch.bmm(x.permute(0, 2, 1), x).sum(dim=0)
            ) / (n_cur_matrix + x.shape[0] * x.shape[1])
            n_cur_matrix += x.shape[0] * x.shape[1]
        elif x.ndim == 2:
            cur_matrix = (cur_matrix * n_cur_matrix + x.T @ x) / (
                n_cur_matrix + x.shape[0]
            )
            n_cur_matrix += x.shape[0]
        else:
            raise ValueError(f"Unsupported input shape {x.shape} for module {name}")

        self._module_inputs[name] = cur_matrix, n_cur_matrix

    def __enter__(self) -> "CaptureModuleInputs":
        # Register hooks for each module to capture inputs and initialize the input matrices
        for name in self._module_names:
            submodule = self._module.get_submodule(name)
            if not isinstance(submodule, nn.Linear):
                raise ValueError(
                    f"Module {name} is not a linear layer, got {type(submodule)}"
                )

            # Initialize the input matrix for the module
            dim = submodule.in_features
            self._module_inputs[name] = torch.zeros((dim, dim), device=self._device), 0

            # Add a forward hook to record the inputs
            handle = submodule.register_forward_hook(partial(self._record, name=name))
            self._handles.append(handle)
        return self

    def __exit__(self, *args: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def module_inputs(self) -> Dict[str, Tensor]:
        """Return the recorded module inputs."""
        return {
            name: input_tensor
            for name, (input_tensor, _) in self._module_inputs.items()
        }


@torch.no_grad()
def sample_module_inputs(
    module: nn.Module,
    module_names: Sequence[str],
    loader: torch.utils.data.DataLoader[Tuple[Tensor, ...]],
    device: torch.device,
) -> Dict[str, Tensor]:
    r"""Sample the internal inputs :math:`\mathbf{H}` [liang24]_ for each module given an external input.

    * Is equivalent to the representation matrix :math:`\textbf{R}` in [saha20]_.

    :param module: Root module to perform forward passes with.
    :param module_names: Submodules to record internal inputs to. Should be LoRA linear or linear layers.
    :param loader: Loader to sample from.
    :return: Return a map between module names and tensors of shape (n_samples, in_features).
    """
    with CaptureModuleInputs(module, module_names, device) as capture:
        for batch in loader:
            x: Tensor = batch[0].to(device)
            module(x)
    return capture.module_inputs


@torch.no_grad()
def create_gpm(
    module_inputs: Dict[str, Tensor],
    threshold: float,
) -> Dict[str, Tensor]:
    r"""Create the gradient projection memory :math:`\mathbf{M}` for the initial task.

    See Section 5.1 of [saha20]_ for an explanation.

    :param module_inputs: Input matrices :math:`\mathbf{H}` [liang24]_ generated by
        :func:`sample_module_inputs`.
    :param threshold: Threshold for k-rank approximation of :math:`\mathbf{H}` (Epsilon
        from [saha20]_ Equation 5).
    :return: Return a map between module names and gradient projection matrix of shape
        (in_features, k)
    """

    # https://github.com/sahagobinda/GPM/blob/1a238ec9d2ca30bae8fd1707f161cc6bd093c72a/main_cifar100.py#L233
    def _f(input: Tensor) -> Tensor:
        # The original implementation uses numpy with double precision.
        # We use torch with float64 for the same effect.
        # TODO: I'm not sure if this is necessary
        U, S, _ = torch.linalg.svd(input.T.to(torch.float64), full_matrices=False)
        # Equation 5 [saha20]_.
        sval_total = (S**2).sum()
        sval_ratio = (S**2) / sval_total
        k = int((torch.cumsum(sval_ratio, 0) < threshold).sum().item())
        return U[:, 0 : max(k, 1)].to(torch.float32)

    return {k: _f(v) for k, v in module_inputs.items()}


@torch.no_grad()
def create_dual_gpm(
    module_inputs: Dict[str, Tensor],
    threshold: float,
) -> Tuple[Dict[str, Tensor], Dict[str, _ProjectType]]:
    def _f(input: Tensor) -> Tuple[Tensor, _ProjectType]:
        # The original implementation uses numpy with double precision.
        # We use torch with float64 for the same effect.
        # TODO: I'm not sure if this is necessary
        U, S, _ = torch.linalg.svd(input.T.to(torch.float64), full_matrices=False)
        # Equation 5 [saha20]_.
        sval_total = (S**2).sum()
        sval_ratio = (S**2) / sval_total
        k = int((torch.cumsum(sval_ratio, 0) < threshold).sum().item())
        gpm = U[:, 0 : max(k, 1)].to(torch.float32)

        if k < input.shape[1] / 2:
            return gpm, _ProjectType.REMOVE
        else:
            return gpm, _ProjectType.RETAIN

    gpm = {}
    project_types = {}
    for k, v in module_inputs.items():
        gpm[k], project_types[k] = _f(v)
    return gpm, project_types


@torch.no_grad()
def update_gpm(
    gpm: Optional[Dict[str, Tensor]],
    module_inputs: Dict[str, Tensor],
    threshold: float,
) -> Dict[str, Tensor]:
    r"""Update the gradient projection matrix :math:`\mathbf{M}` for subsequent tasks.

    See Section 5.2 of [saha20]_ for an explanation.

    :param gpm: Gradient projection memory :math:`\mathbf{M}`.
    :param module_inputs: Input matrices :math:`\mathbf{H}`.
    :param threshold: Threshold for k-rank approximation of :math:`\mathbf{H}` (Epsilon
        from [saha20]_ Equation 5).
    """
    if gpm is None:
        return create_gpm(module_inputs, threshold)

    # https://github.com/sahagobinda/GPM/blob/1a238ec9d2ca30bae8fd1707f161cc6bd093c72a/main_cifar100.py#L242
    def _f(input: Tensor, gpm: Tensor) -> Tensor:
        inputs_hat = _get_input_hat(input, gpm)
        u, s, _ = torch.linalg.svd(inputs_hat.to(torch.float64).T, full_matrices=False)

        # Equation 9 [saha20]_.
        _, s_inputs, _ = torch.linalg.svd(input.to(torch.float64).T, full_matrices=True)
        sval_total = (s_inputs**2).sum()
        sval_hat = (s**2).sum()
        sval_ratio = (s**2) / sval_total
        accumulated_sval = (sval_total - sval_hat) / sval_total

        r = 0
        for ii in range(sval_ratio.shape[0]):
            if accumulated_sval < threshold:
                accumulated_sval += sval_ratio[ii]
                r += 1
            else:
                break
        if r == 0:
            return gpm

        gpm_new = torch.hstack((gpm, u[:, 0:r])).float()
        if u.shape[1] > u.shape[0]:
            return gpm_new[:, 0 : gpm_new.shape[0]]
        else:
            return gpm_new

    keys = module_inputs.keys() & gpm.keys()
    if not (len(keys) == len(gpm.keys()) == len(module_inputs.keys())):
        raise ValueError("``module_inputs`` and ``gpm`` contain different keys")
    return {k: _f(module_inputs[k], gpm[k]) for k in keys}


def orthogonal_projection_matrix(
    module_inputs: Dict[str, Tensor],
    gpm: Optional[Dict[str, Tensor]],
    rank: Dict[str, int],
) -> Dict[str, Tensor]:
    r"""Compute an orthogonal projection matrix for use in LoRA's A matrix.

    :module inputs: Input matrices :math:`\mathbf{H}` [liang24]_ generated by
        :func:`sample_module_inputs` from a new task before training.
    :param gpm: Gradient projection memory :math:`\mathbf{M}` from previous tasks.
    :param rank: Rank of the LoRA A matrix for each module.
    :return: Return a map between module names and orthogonal projection matrices of shape
        (rank, in_features) for each module.
    """

    def _opm(input: Tensor, rank: int) -> Tensor:
        """Orthogonal projection matrix for LoRA's A matrix."""
        u, _, _ = torch.linalg.svd(input.T, full_matrices=False)
        # TODO: I'm not sure why we are scaling by ``sqrt(3)`` but it is in the original
        # paper.
        return (u[:, :rank] / math.sqrt(3)).float().T

    def _opm_gpm(input: Tensor, gpm: Tensor, rank: int) -> Tensor:
        """Orthogonal projection matrix for LoRA's A matrix with GPM."""
        return _opm(_get_input_hat(input, gpm), rank)

    if gpm is not None:
        return {k: _opm_gpm(module_inputs[k], gpm[k], rank[k]) for k in module_inputs}
    else:
        return {k: _opm(module_inputs[k], rank[k]) for k in module_inputs}
