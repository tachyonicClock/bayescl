import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def matrix_normal_sample_cholesky(
    M: Tensor, L_u: Tensor, L_v: Tensor, Z: Tensor | None = None
) -> Tensor:
    """Sample from a matrix normal distribution with mean M and Cholesky factors L_u and L_v.

    :param M: The mean matrix of shape (n, p).
    :param L_u: The lower-triangular Cholesky factor of the row covariance matrix of shape (n, n).
    :param L_v: The lower-triangular Cholesky factor of the column covariance matrix of shape (p, p).
    :param Z: Optional standard-normal noise matrix of shape (n, p). If not provided,
        a fresh standard-normal draw is used.
    :return: A sample from the matrix normal distribution of shape (n, p).
    """
    n, p = M.shape
    assert L_u.shape == (n, n)
    assert L_v.shape == (p, p)

    if Z is None:
        Z = torch.randn_like(M)
    else:
        assert Z.shape == (n, p)
    return M + L_u @ Z @ L_v.T


def matrix_normal_sample(
    M: Tensor, U: Tensor, V: Tensor, Z: Tensor | None = None
) -> Tensor:
    """Sample from a matrix normal distribution with mean M and covariance matrices U and V.

    :param M: The mean matrix of shape (n, p).
    :param U: The row covariance matrix of shape (n, n) (positive definite).
    :param V: The column covariance matrix of shape (p, p) (positive definite).
    :param Z: Optional standard-normal noise matrix of shape (n, p). If not provided,
        a fresh standard-normal draw is used.
    :return: A sample from the matrix normal distribution of shape (n, p).
    """
    n, p = M.shape
    assert U.shape == (n, n)
    assert V.shape == (p, p)

    return matrix_normal_sample_cholesky(
        M,
        torch.linalg.cholesky(U),
        torch.linalg.cholesky(V),
        Z,
    )


def matrix_normal_kl(
    M_1: Tensor,
    U_1: Tensor,
    V_1: Tensor,
    M_2: Tensor,
    U_2: Tensor,
    V_2: Tensor,
) -> Tensor:
    r"""KL divergence between two matrix normal distributions.

    $$
    \begin{split}
        \mathrm{KL}[P\,||\,Q] = \frac{1}{2} \Big[
            % term 1 
            &\mathrm{vec}(M_2 - M_1)^\mathrm{T} \mathrm{vec}\left(U_2^{-1} (M_2 - M_1) V_2^{-1}\right) \\
            % term 2
            &+ \mathrm{tr}\left( (V_2^{-1}V_1) \otimes (U_2^{-1}U_1) \right) \\
            % term 3
            &- n \ln \frac{|V_1|}{|V_2|} - p \ln \frac{|U_1|}{|U_2|} - np 
        \Big] \; .
    \end{split}
    $$

    https://statproofbook.github.io/P/matn-kl.html

    :param M_1: Mean of P, shape (n, p).
    :param U_1: Row covariance of P, shape (n, n) (positive definite).
    :param V_1: Column covariance of P, shape (p, p) (positive definite).
    :param M_2: Mean of Q, shape (n, p).
    :param U_2: Row covariance of Q, shape (n, n) (positive definite).
    :param V_2: Column covariance of Q, shape (p, p) (positive definite).
    :return: KL[P || Q] as a scalar tensor.
    """
    n, p = M_1.shape
    assert M_2.shape == (n, p)
    assert U_1.shape == (n, n)
    assert U_2.shape == (n, n)
    assert V_1.shape == (p, p)
    assert V_2.shape == (p, p)

    return matrix_normal_kl_cholesky(
        M_1,
        torch.linalg.cholesky(U_1),
        torch.linalg.cholesky(V_1),
        M_2,
        torch.linalg.cholesky(U_2),
        torch.linalg.cholesky(V_2),
    )


def matrix_normal_kl_cholesky(
    M_1: Tensor,
    L_u1: Tensor,
    L_v1: Tensor,
    M_2: Tensor,
    L_u2: Tensor,
    L_v2: Tensor,
) -> Tensor:
    r"""KL divergence between two matrix normal distributions given their Cholesky factors.

    More efficient than :func:`matrix_normal_kl` because it avoids re-factorizing the
    covariance matrices and replaces general solves with cheaper triangular solves.

    $$
    \begin{split}
        \mathrm{KL}[P\,||\,Q] = \frac{1}{2} \Big[
            % term 1 
            &\mathrm{vec}(M_2 - M_1)^\mathrm{T} \mathrm{vec}\left(U_2^{-1} (M_2 - M_1) V_2^{-1}\right) \\
            % term 2
            &+ \mathrm{tr}\left( (V_2^{-1}V_1) \otimes (U_2^{-1}U_1) \right) \\
            % term 3
            &- n \ln \frac{|V_1|}{|V_2|} \\
            % term 4
            &- p \ln \frac{|U_1|}{|U_2|} \\
            % term 5
            &- np 
        \Big] \; .
    \end{split}
    $$

    :param M_1: Mean of P, shape (n, p).
    :param L_u1: Lower-triangular Cholesky factor of U_1, shape (n, n).
    :param L_v1: Lower-triangular Cholesky factor of V_1, shape (p, p).
    :param M_2: Mean of Q, shape (n, p).
    :param L_u2: Lower-triangular Cholesky factor of U_2, shape (n, n).
    :param L_v2: Lower-triangular Cholesky factor of V_2, shape (p, p).
    :return: KL[P || Q] as a scalar tensor.
    """
    n, p = M_1.shape
    M_Delta = M_2 - M_1

    # Term 1
    r"""
    We begin by simplifying the first term,
    $$
    \text{vec}(M_2 - M_1)^\text{T} \text{vec}\left(U_2^{-1} (M_2 - M_1) V_2^{-1}\right).
    $$

    By defining the difference matrix $M_\Delta = M_2 - M_1$ and applying the identity
    $\text{vec}(A)^\text{T} \text{vec}(B) = \text{tr}(A^\text{T} B)$, the expression
    becomes
    $$
    \text{tr}(M_\Delta^\text{T} U_2^{-1} M_\Delta V_2^{-1}).
    $$
    
    Substituting the Cholesky factorizations $U_2 = L_{U2} L_{U2}^\text{T}$ and
    $V_2 = L_{V2} L_{V2}^\text{T}$ into the trace yields
    $$
    \text{tr}(
        M_\Delta^\text{T} L_{U2}^{-\text{T}} L_{U2}^{-1}
        M_\Delta L_{V2}^{-\text{T}} L_{V2}^{-1}
    ),
    $$
    which can be rearranged by invoking the cyclic property of the trace to obtain
    $$
    \text{tr}(
        (L_{U2}^{-1} M_\Delta L_{V2}^{-\text{T}})^\text{T}
        (L_{U2}^{-1} M_\Delta L_{V2}^{-\text{T}})
    ).
    $$

    By substituting $X = L_{U2}^{-1} M_\Delta L_{V2}^{-\text{T}}$, the expression
    simplifies to the inner product form
    $$
    \text{tr}(X^\text{T} X),
    $$
    which, by the definition of the Frobenius norm, is equivalent to
    $$
    \|X\|_F^2 = \sum_{i,j} X_{ij}^2.
    $$
    """
    # Compute the intermediate matrix $X = L_u2^{-1} M_Delta L_v2^{-T}$ using triangular
    # solves.
    A = torch.linalg.solve_triangular(L_u2, M_Delta, upper=False)
    B = torch.linalg.solve_triangular(L_v2, A.mT, upper=False)
    term1 = torch.pow(B, 2).sum()

    # Term 2
    S_u = torch.linalg.solve_triangular(L_u2, L_u1, upper=False)
    S_v = torch.linalg.solve_triangular(L_v2, L_v1, upper=False)
    term2 = torch.pow(S_u, 2).sum() * torch.pow(S_v, 2).sum()

    # Term 3 & 4: Log-determinants
    # log |X| = 2 * sum(log(diag(L_x)))
    log_det_U1 = 2.0 * L_u1.diagonal(dim1=-2, dim2=-1).log().sum()
    log_det_U2 = 2.0 * L_u2.diagonal(dim1=-2, dim2=-1).log().sum()
    log_det_V1 = 2.0 * L_v1.diagonal(dim1=-2, dim2=-1).log().sum()
    log_det_V2 = 2.0 * L_v2.diagonal(dim1=-2, dim2=-1).log().sum()

    # Following the grouping: -n ln(|V1|/|V2|) = n (ln|V2| - ln|V1|)
    term3 = n * (log_det_V2 - log_det_V1)
    term4 = p * (log_det_U2 - log_det_U1)
    term5 = -n * p

    return 0.5 * (term1 + term2 + term3 + term4 + term5)


class MatrixNormalDistribution(nn.Module):
    def __init__(self, n: int, p: int, init_std: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.n = n
        self.p = p
        self.init_std = init_std
        self.eps = eps

        self.M = nn.Parameter(torch.zeros(n, p))

        self.L_u_raw = nn.Parameter(torch.zeros(n, n))
        self.L_v_raw = nn.Parameter(torch.zeros(p, p))

        inv_softplus_std = math.log(math.exp(self.init_std) - 1)
        nn.init.constant_(self.L_u_raw.diagonal(), inv_softplus_std)
        nn.init.constant_(self.L_v_raw.diagonal(), inv_softplus_std)

    def _get_valid_cholesky(self, L_raw: Tensor) -> Tensor:
        """Transforms unconstrained parameters into a valid Cholesky factor."""
        # Force lower triangular (zeros out the upper triangle)
        L = torch.tril(L_raw, diagonal=-1)

        # Force the diagonal to be strictly positive using softplus + epsilon
        diag = F.softplus(L_raw.diagonal()) + self.eps
        L = L + torch.diag_embed(diag)
        return L

    @property
    def L_u(self) -> Tensor:
        return self._get_valid_cholesky(self.L_u_raw)

    @property
    def L_v(self) -> Tensor:
        return self._get_valid_cholesky(self.L_v_raw)

    def forward(self) -> Tensor:
        return matrix_normal_sample_cholesky(self.M, self.L_u, self.L_v)

    @property
    def U(self) -> Tensor:
        L = self.L_u
        return L @ L.T

    @property
    def V(self) -> Tensor:
        L = self.L_v
        return L @ L.T

    def kl_divergence(self, other: "MatrixNormalDistribution") -> Tensor:
        return matrix_normal_kl_cholesky(
            self.M,
            self.L_u,
            self.L_v,
            other.M,
            other.L_u,
            other.L_v,
        )
