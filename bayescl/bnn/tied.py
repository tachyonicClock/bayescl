"""Forward passes and KL divergences for a tied Bayesian LoRA.

The variational posterior over each column of ``A`` and each row of ``B`` is a
multivariate normal with a covariance ``S = L L^T`` that is shared across all
columns and rows. All functions accept the lower-triangular Cholesky factor
``L`` with a positive diagonal instead of ``S``.
"""

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, kl_divergence


def forward_tied_lora(A: Tensor, B: Tensor, L: Tensor, x: Tensor) -> Tensor:
    r"""Forward pass through the tied Bayesian LoRA.

    :param A: Mean matrix of the ``A`` distribution ``(rank_dim, in_dim)``.
    :param B: Mean matrix of the ``B`` distribution ``(out_dim, rank_dim)``.
    :param L: Lower-triangular Cholesky factor, with a positive diagonal, of the
        covariance :math:`S = LL^\top` shared across ranks. ``(rank_dim, rank_dim)``.
    :param x: Batch of input data ``(batch_dim, in_dim)``.
    :return: Batch of output data ``(batch_dim, out_dim)``.
    """
    device, dtype = A.device, A.dtype
    rank_dim, in_dim = A.shape
    out_dim, B_rank_dim = B.shape
    batch_dim, x_in_dim = x.shape

    # Check shape, device, and dtype
    if B_rank_dim != rank_dim:
        raise ValueError(f"B has rank_dim {B_rank_dim}, expected {rank_dim}")
    if L.shape != (rank_dim, rank_dim):
        raise ValueError(
            f"L has shape {tuple(L.shape)}, expected {(rank_dim, rank_dim)}"
        )
    if x_in_dim != in_dim:
        raise ValueError(f"x has in_dim {x_in_dim}, expected {in_dim}")
    if not device == B.device == L.device == x.device:
        raise ValueError("A, B, L, and x must be on the same device")
    if not dtype == B.dtype == L.dtype == x.dtype:
        raise ValueError("A, B, L, and x must have the same dtype")

    # Sample A: one (rank_dim,) draw per batch element per column, tied via L
    A_samples = torch.empty(batch_dim, rank_dim, in_dim, device=device, dtype=dtype)
    for j in range(in_dim):
        q_A = MultivariateNormal(A[:, j], scale_tril=L)
        A_samples[:, :, j] = q_A.rsample((batch_dim,))  # (batch_dim, rank_dim)

    # Sample B: one (rank_dim,) draw per batch element per row, tied via L
    B_samples = torch.empty(batch_dim, out_dim, rank_dim, device=device, dtype=dtype)
    for k in range(out_dim):
        q_B = MultivariateNormal(B[k, :], scale_tril=L)
        B_samples[:, k, :] = q_B.rsample((batch_dim,))  # (batch_dim, rank_dim)

    # delta_W x = B (A x), computed per batch element without forming delta_W
    Ax = A_samples @ x.unsqueeze(-1)  # (batch_dim, rank_dim, 1)
    out = (B_samples @ Ax).squeeze(-1)  # (batch_dim, out_dim)

    return out


def forward_tied_lora_fast(A: Tensor, B: Tensor, L: Tensor, x: Tensor) -> Tensor:
    r"""Forward pass through the tied Bayesian LoRA.

    Distributionally equivalent to :func:`forward_tied_lora`, but never
    materializes the sampled ``A`` or ``B`` matrices. Because a linear map of a
    Gaussian is Gaussian, the projected noise is sampled directly:

    .. math::

        \epsilon_A x \sim \mathcal{N}(0, \|x\|^2 I_r), \qquad
        \epsilon_B L^\top A x \sim \mathcal{N}(0, \|L^\top A x\|^2 I_m)

    Memory scales as ``O(batch_dim * (rank_dim + out_dim))``. This function uses
    a different random stream than :func:`forward_tied_lora`, so compare the two
    by moments, not by individual samples.

    :param A: Mean matrix of the ``A`` distribution ``(rank_dim, in_dim)``.
    :param B: Mean matrix of the ``B`` distribution ``(out_dim, rank_dim)``.
    :param L: Lower-triangular Cholesky factor, with a positive diagonal, of the
        covariance :math:`S = LL^\top` shared across ranks. ``(rank_dim, rank_dim)``.
    :param x: Batch of input data ``(batch_dim, in_dim)``.
    :return: Batch of output data ``(batch_dim, out_dim)``.
    """
    device, dtype = A.device, A.dtype
    rank_dim, in_dim = A.shape
    out_dim, B_rank_dim = B.shape
    batch_dim, x_in_dim = x.shape

    # Check shape, device, and dtype
    if B_rank_dim != rank_dim:
        raise ValueError(f"B has rank_dim {B_rank_dim}, expected {rank_dim}")
    if L.shape != (rank_dim, rank_dim):
        raise ValueError(
            f"L has shape {tuple(L.shape)}, expected {(rank_dim, rank_dim)}"
        )
    if x_in_dim != in_dim:
        raise ValueError(f"x has in_dim {x_in_dim}, expected {in_dim}")
    if not device == B.device == L.device == x.device:
        raise ValueError("A, B, L, and x must be on the same device")
    if not dtype == B.dtype == L.dtype == x.dtype:
        raise ValueError("A, B, L, and x must have the same dtype")

    # A x = A_mean x + L (eps_A x), where eps_A x ~ N(0, ||x||^2 I)
    x_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)  # (batch_dim, 1)
    z_A = torch.randn(batch_dim, rank_dim, device=device, dtype=dtype)
    Ax = x @ A.T + x_norm * (z_A @ L.T)  # (batch_dim, rank_dim)

    # B (A x) = B_mean (A x) + eps_B (L^T A x), where the noise ~ N(0, ||L^T A x||^2 I)
    LtAx_norm = torch.linalg.vector_norm(Ax @ L, dim=-1, keepdim=True)  # (batch_dim, 1)
    z_B = torch.randn(batch_dim, out_dim, device=device, dtype=dtype)
    out = Ax @ B.T + LtAx_norm * z_B  # (batch_dim, out_dim)

    return out


def sample_tied_lora_weights(A: Tensor, B: Tensor, L: Tensor) -> tuple[Tensor, Tensor]:
    r"""Draw one ``(A, B)`` pair from the tied posterior.

    The counterpart to :func:`forward_tied_lora_fast`, which draws independent
    weights per batch element. Here a single pair is drawn and the caller
    applies it to a whole batch, so the noise is ``O(rank_dim * (in_dim +
    out_dim))`` rather than ``O(batch_dim * (rank_dim + out_dim))``. That is a
    higher-variance gradient estimator but a much cheaper one once
    ``batch_dim`` exceeds the adapter's own size.

    Each column of ``A`` and each row of ``B`` is drawn from the shared
    covariance :math:`S = LL^\top`:

    .. math::

        A + LZ, \quad Z \sim \mathcal{N}(0, I_{r \times n}), \qquad
        B + WL^\top, \quad W \sim \mathcal{N}(0, I_{m \times r})

    :param A: Mean matrix of the ``A`` distribution ``(rank_dim, in_dim)``.
    :param B: Mean matrix of the ``B`` distribution ``(out_dim, rank_dim)``.
    :param L: Lower-triangular Cholesky factor, with a positive diagonal, of the
        covariance :math:`S = LL^\top` shared across ranks. ``(rank_dim, rank_dim)``.
    :return: A sampled ``(A, B)`` pair with the same shapes as the means.
    """
    device, dtype = A.device, A.dtype
    rank_dim, _ = A.shape
    out_dim, B_rank_dim = B.shape

    # Check shape, device, and dtype
    if B_rank_dim != rank_dim:
        raise ValueError(f"B has rank_dim {B_rank_dim}, expected {rank_dim}")
    if L.shape != (rank_dim, rank_dim):
        raise ValueError(
            f"L has shape {tuple(L.shape)}, expected {(rank_dim, rank_dim)}"
        )
    if not device == B.device == L.device:
        raise ValueError("A, B, and L must be on the same device")
    if not dtype == B.dtype == L.dtype:
        raise ValueError("A, B, and L must have the same dtype")

    z_A = torch.randn_like(A)  # (rank_dim, in_dim)
    z_B = torch.randn(out_dim, rank_dim, device=device, dtype=dtype)
    return A + L @ z_A, B + z_B @ L.T


def kl_divergence_tied_lora(
    A: Tensor, B: Tensor, L: Tensor, prior_A: Tensor, prior_B: Tensor, prior_L: Tensor
) -> Tensor:
    r"""Compute the KL divergence between LoRA modules with tied covariances.

    :param A: Mean matrix of the ``A`` distribution ``(rank_dim, in_dim)``.
    :param B: Mean matrix of the ``B`` distribution ``(out_dim, rank_dim)``.
    :param L: Lower-triangular Cholesky factor, with a positive diagonal, of the
        covariance :math:`S = LL^\top` shared across ranks. ``(rank_dim, rank_dim)``.
    :param prior_A: Prior mean for ``A`` ``(rank_dim, in_dim)``.
    :param prior_B: Prior mean for ``B`` ``(out_dim, rank_dim)``.
    :param prior_L: Lower-triangular Cholesky factor of the prior covariance,
        shared across ranks (typically isotropic, for example
        ``sigma_p * I``). ``(rank_dim, rank_dim)``.
    :return: Unscaled scalar KL tensor.
    """
    device, dtype = A.device, A.dtype
    rank_dim, in_dim = A.shape
    out_dim, B_rank_dim = B.shape

    # Check shape, device, and dtype
    if B_rank_dim != rank_dim:
        raise ValueError(f"B has rank_dim {B_rank_dim}, expected {rank_dim}")
    if prior_A.shape != A.shape:
        raise ValueError(
            f"prior_A has shape {tuple(prior_A.shape)}, expected {tuple(A.shape)}"
        )
    if prior_B.shape != B.shape:
        raise ValueError(
            f"prior_B has shape {tuple(prior_B.shape)}, expected {tuple(B.shape)}"
        )
    if L.shape != (rank_dim, rank_dim) or prior_L.shape != (rank_dim, rank_dim):
        raise ValueError(f"L and prior_L must have shape {(rank_dim, rank_dim)}")
    if (
        not device
        == B.device
        == L.device
        == prior_A.device
        == prior_B.device
        == prior_L.device
    ):
        raise ValueError("All inputs must be on the same device")
    if (
        not dtype
        == B.dtype
        == L.dtype
        == prior_A.dtype
        == prior_B.dtype
        == prior_L.dtype
    ):
        raise ValueError("All inputs must have the same dtype")

    # Sum KL divergence over each column of A and each row of B, tied via L
    terms = []
    for j in range(in_dim):
        q_A = MultivariateNormal(A[:, j], scale_tril=L)
        p_A = MultivariateNormal(prior_A[:, j], scale_tril=prior_L)
        terms.append(kl_divergence(q_A, p_A))

    for k in range(out_dim):
        q_B = MultivariateNormal(B[k, :], scale_tril=L)
        p_B = MultivariateNormal(prior_B[k, :], scale_tril=prior_L)
        terms.append(kl_divergence(q_B, p_B))

    return torch.stack(terms).sum()


def kl_divergence_tied_lora_fast(
    A: Tensor, B: Tensor, L: Tensor, prior_A: Tensor, prior_B: Tensor, prior_L: Tensor
) -> Tensor:
    r"""Compute the KL divergence between LoRA modules with tied covariances.

    Closed-form, vectorized equivalent of :func:`kl_divergence_tied_lora`. Sums
    the per-column and per-row Gaussian KL terms analytically, using triangular
    solves against :math:`L_p` instead of matrix inverses:

    .. math::

        D_{KL} = \tfrac{1}{2}(n+m)\left[\|L_p^{-1}L\|_F^2 - r +
            2\sum_i \ln (L_p)_{ii} - 2\sum_i \ln L_{ii}\right]
            + \tfrac{1}{2}\left[\|L_p^{-1}\Delta_A\|_F^2
            + \|L_p^{-1}\Delta_B^\top\|_F^2\right]

    where :math:`\Delta_A = A - \text{prior\_A}`, :math:`\Delta_B = B - \text{prior\_B}`,
    :math:`n` = ``in_dim``, :math:`m` = ``out_dim``, and :math:`r` = ``rank_dim``.

    :param A: Mean matrix of the ``A`` distribution ``(rank_dim, in_dim)``.
    :param B: Mean matrix of the ``B`` distribution ``(out_dim, rank_dim)``.
    :param L: Lower-triangular Cholesky factor, with a positive diagonal, of the
        covariance :math:`S = LL^\top` shared across ranks. ``(rank_dim, rank_dim)``.
    :param prior_A: Prior mean for ``A`` ``(rank_dim, in_dim)``.
    :param prior_B: Prior mean for ``B`` ``(out_dim, rank_dim)``.
    :param prior_L: Lower-triangular Cholesky factor of the prior covariance,
        shared across ranks (typically isotropic, for example
        ``sigma_p * I``). ``(rank_dim, rank_dim)``.
    :return: Unscaled scalar KL tensor.
    """
    device, dtype = A.device, A.dtype
    rank_dim, in_dim = A.shape
    out_dim, B_rank_dim = B.shape

    # Check shape, device, and dtype
    if B_rank_dim != rank_dim:
        raise ValueError(f"B has rank_dim {B_rank_dim}, expected {rank_dim}")
    if prior_A.shape != A.shape:
        raise ValueError(
            f"prior_A has shape {tuple(prior_A.shape)}, expected {tuple(A.shape)}"
        )
    if prior_B.shape != B.shape:
        raise ValueError(
            f"prior_B has shape {tuple(prior_B.shape)}, expected {tuple(B.shape)}"
        )
    if L.shape != (rank_dim, rank_dim) or prior_L.shape != (rank_dim, rank_dim):
        raise ValueError(f"L and prior_L must have shape {(rank_dim, rank_dim)}")
    if (
        not device
        == B.device
        == L.device
        == prior_A.device
        == prior_B.device
        == prior_L.device
    ):
        raise ValueError("All inputs must be on the same device")
    if (
        not dtype
        == B.dtype
        == L.dtype
        == prior_A.dtype
        == prior_B.dtype
        == prior_L.dtype
    ):
        raise ValueError("All inputs must have the same dtype")

    # Log-determinant term: ln det S = 2 * sum(ln diag L)
    logdet_term = 2 * (prior_L.diagonal().log().sum() - L.diagonal().log().sum())

    # Trace term: tr(S_p^{-1} S) = ||L_p^{-1} L||_F^2
    M = torch.linalg.solve_triangular(prior_L, L, upper=False)
    trace_term = M.square().sum()

    bracket = trace_term - rank_dim + logdet_term

    # Quadratic mean-shift term, summed over all columns of A and rows of B at once
    delta_A = torch.linalg.solve_triangular(prior_L, A - prior_A, upper=False)
    delta_B = torch.linalg.solve_triangular(prior_L, (B - prior_B).T, upper=False)
    quad_term = delta_A.square().sum() + delta_B.square().sum()

    kl = 0.5 * (in_dim + out_dim) * bracket + 0.5 * quad_term

    return kl
