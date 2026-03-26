import pytest
import torch
from mnd import (
    MatrixNormalDistribution,
    matrix_normal_kl,
    matrix_normal_kl_cholesky,
    matrix_normal_sample,
)
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence


def _spd_from_seed(size: int, seed: int, jitter: float = 0.5) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(size, size, generator=g, dtype=torch.float64)
    return a @ a.mT + jitter * torch.eye(size, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_distribution_params():
    """Shared parameters for KL-divergence tests involving two distinct distributions."""
    torch.set_default_dtype(torch.float64)
    n, p = 3, 2
    M_1 = torch.tensor([[0.2, -0.5], [1.0, 0.3], [-0.2, 0.7]], dtype=torch.float64)
    M_2 = torch.tensor([[-0.1, 0.4], [0.5, -0.9], [0.6, 0.2]], dtype=torch.float64)
    U_1 = _spd_from_seed(n, seed=50)
    U_2 = _spd_from_seed(n, seed=60)
    V_1 = _spd_from_seed(p, seed=70)
    V_2 = _spd_from_seed(p, seed=80)
    return n, p, M_1, M_2, U_1, U_2, V_1, V_2


@pytest.fixture
def identical_distribution_params():
    """Shared parameters for KL=0 tests (identical distributions)."""
    torch.set_default_dtype(torch.float64)
    n, p = 2, 2
    M = torch.tensor([[0.4, -0.2], [0.9, 0.3]], dtype=torch.float64)
    U = _spd_from_seed(n, seed=90)
    V = _spd_from_seed(p, seed=100)
    return n, p, M, U, V


# ---------------------------------------------------------------------------
# Sampling tests
# ---------------------------------------------------------------------------


def test_matrix_normal_sample_matches_mvn_with_shared_noise():
    """Verify X = M + L_u ε L_v^T equals the equivalent Kronecker-product MVN transform.

    A fixed noise matrix ε is applied via both representations to test the
    algebraic identity without relying on the RNG being consumed in the same
    order by both code paths.
    """
    torch.set_default_dtype(torch.float64)

    n, p = 3, 2
    M = torch.tensor([[0.2, -0.3], [1.1, 0.4], [-0.5, 0.7]], dtype=torch.float64)
    U = _spd_from_seed(n, seed=10)
    V = _spd_from_seed(p, seed=20)

    L_u = torch.linalg.cholesky(U)
    L_v = torch.linalg.cholesky(V)

    # Draw a single fixed noise matrix and apply the transformation explicitly,
    # avoiding reliance on RNG state being consumed identically by both paths.
    epsilon = torch.randn(
        n, p, dtype=torch.float64, generator=torch.Generator().manual_seed(1234)
    )

    # Matrix Normal sample using explicit noise, so we test matrix_normal_sample directly.
    sample_matrix = matrix_normal_sample(M, U, V, Z=epsilon).reshape(-1)

    # Equivalent MVN transformation: vec(X) = vec(M) + kron(L_u, L_v) @ vec(ε)
    scale_tril = torch.kron(L_u, L_v)
    sample_mvn = M.reshape(-1) + scale_tril @ epsilon.reshape(-1)

    assert torch.allclose(sample_matrix, sample_mvn, atol=1e-10, rtol=1e-10)


def test_matrix_normal_sample_empirical_moments_match_analytical():
    """Statistical sanity check: empirical mean and covariance should match the
    analytical values M and kron(U, V) to within Monte Carlo error.

    The fixed seed prevents CI flakiness.  Since
    test_matrix_normal_sample_matches_mvn_with_shared_noise already verifies
    the algebraic identity, this test serves as a distributional sanity check.
    """
    torch.set_default_dtype(torch.float64)

    n, p = 2, 3
    M = torch.tensor([[0.3, -0.2, 0.1], [1.0, 0.4, -0.7]], dtype=torch.float64)
    U = _spd_from_seed(n, seed=30)
    V = _spd_from_seed(p, seed=40)

    num_samples = 12_000
    torch.manual_seed(2026)
    samples = torch.stack(
        [matrix_normal_sample(M, U, V).reshape(-1) for _ in range(num_samples)], dim=0
    )

    centered = samples - samples.mean(dim=0)
    empirical_cov = centered.mT @ centered / (num_samples - 1)

    assert torch.allclose(samples.mean(dim=0), M.reshape(-1), atol=8e-2, rtol=0.0)
    assert torch.allclose(empirical_cov, torch.kron(U, V), atol=2e-1, rtol=0.0)


# ---------------------------------------------------------------------------
# KL divergence tests
# ---------------------------------------------------------------------------


def test_matrix_normal_kl_matches_mvn_kl(two_distribution_params):
    n, p, M_1, M_2, U_1, U_2, V_1, V_2 = two_distribution_params

    kl_matrix_normal = matrix_normal_kl(M_1, U_1, V_1, M_2, U_2, V_2)

    p_dist = MultivariateNormal(
        loc=M_1.reshape(-1), covariance_matrix=torch.kron(U_1, V_1)
    )
    q_dist = MultivariateNormal(
        loc=M_2.reshape(-1), covariance_matrix=torch.kron(U_2, V_2)
    )
    kl_mvn = kl_divergence(p_dist, q_dist)

    assert torch.allclose(kl_matrix_normal, kl_mvn, atol=1e-8, rtol=1e-8)


def test_matrix_normal_kl_is_zero_for_identical_distributions(
    identical_distribution_params,
):
    _, _, M, U, V = identical_distribution_params

    kl_val = matrix_normal_kl(M, U, V, M, U, V)
    assert torch.allclose(
        kl_val, torch.tensor(0.0, dtype=torch.float64), atol=1e-10, rtol=0.0
    )


def test_matrix_normal_distribution_forward_and_covariances_are_valid():
    torch.set_default_dtype(torch.float64)

    n, p = 3, 2
    dist = MatrixNormalDistribution(n=n, p=p, init_std=0.7)

    sample = dist()
    assert sample.shape == (n, p)

    L_u = dist.L_u
    L_v = dist.L_v
    assert torch.allclose(L_u, torch.tril(L_u), atol=1e-12, rtol=0.0)
    assert torch.allclose(L_v, torch.tril(L_v), atol=1e-12, rtol=0.0)
    assert torch.all(torch.diagonal(L_u) > 0)
    assert torch.all(torch.diagonal(L_v) > 0)

    eig_u = torch.linalg.eigvalsh(dist.U)
    eig_v = torch.linalg.eigvalsh(dist.V)
    assert torch.all(eig_u > 0)
    assert torch.all(eig_v > 0)


def test_matrix_normal_kl_cholesky_matches_kl(two_distribution_params):
    """matrix_normal_kl_cholesky must agree with matrix_normal_kl on identical inputs."""
    n, p, M_1, M_2, U_1, U_2, V_1, V_2 = two_distribution_params

    L_u1 = torch.linalg.cholesky(U_1)
    L_u2 = torch.linalg.cholesky(U_2)
    L_v1 = torch.linalg.cholesky(V_1)
    L_v2 = torch.linalg.cholesky(V_2)

    kl_ref = matrix_normal_kl(M_1, U_1, V_1, M_2, U_2, V_2)
    kl_chol = matrix_normal_kl_cholesky(M_1, L_u1, L_v1, M_2, L_u2, L_v2)

    assert torch.allclose(kl_chol, kl_ref, atol=1e-8, rtol=1e-8)


def test_matrix_normal_kl_cholesky_matches_mvn_kl(two_distribution_params):
    """matrix_normal_kl_cholesky must agree with the MVN KL reference."""
    n, p, M_1, M_2, U_1, U_2, V_1, V_2 = two_distribution_params

    L_u1 = torch.linalg.cholesky(U_1)
    L_u2 = torch.linalg.cholesky(U_2)
    L_v1 = torch.linalg.cholesky(V_1)
    L_v2 = torch.linalg.cholesky(V_2)

    kl_chol = matrix_normal_kl_cholesky(M_1, L_u1, L_v1, M_2, L_u2, L_v2)

    p_dist = MultivariateNormal(
        loc=M_1.reshape(-1), covariance_matrix=torch.kron(U_1, V_1)
    )
    q_dist = MultivariateNormal(
        loc=M_2.reshape(-1), covariance_matrix=torch.kron(U_2, V_2)
    )
    kl_mvn = kl_divergence(p_dist, q_dist)

    assert torch.allclose(kl_chol, kl_mvn, atol=1e-8, rtol=1e-8)


def test_matrix_normal_kl_cholesky_is_zero_for_identical_distributions(
    identical_distribution_params,
):
    _, _, M, U, V = identical_distribution_params

    L_u = torch.linalg.cholesky(U)
    L_v = torch.linalg.cholesky(V)

    kl_val = matrix_normal_kl_cholesky(M, L_u, L_v, M, L_u, L_v)
    assert torch.allclose(
        kl_val, torch.tensor(0.0, dtype=torch.float64), atol=1e-10, rtol=0.0
    )


def test_matrix_normal_kl_cholesky_is_non_negative():
    torch.set_default_dtype(torch.float64)

    n, p = 4, 3
    M_1 = torch.randn(
        n, p, dtype=torch.float64, generator=torch.Generator().manual_seed(1)
    )
    M_2 = torch.randn(
        n, p, dtype=torch.float64, generator=torch.Generator().manual_seed(2)
    )
    L_u1 = torch.linalg.cholesky(_spd_from_seed(n, seed=11))
    L_u2 = torch.linalg.cholesky(_spd_from_seed(n, seed=22))
    L_v1 = torch.linalg.cholesky(_spd_from_seed(p, seed=33))
    L_v2 = torch.linalg.cholesky(_spd_from_seed(p, seed=44))

    kl_val = matrix_normal_kl_cholesky(M_1, L_u1, L_v1, M_2, L_u2, L_v2)
    assert kl_val >= 0.0


def test_matrix_normal_distribution_kl_divergence_matches_function():
    torch.set_default_dtype(torch.float64)

    n, p = 2, 3
    q = MatrixNormalDistribution(n=n, p=p, init_std=0.9)
    p_dist = MatrixNormalDistribution(n=n, p=p, init_std=1.1)

    with torch.no_grad():
        p_dist.M.copy_(
            torch.tensor([[0.4, -0.2, 0.1], [0.7, 0.3, -0.5]], dtype=torch.float64)
        )
        q.M.copy_(
            torch.tensor([[-0.1, 0.6, -0.4], [0.2, -0.7, 0.8]], dtype=torch.float64)
        )

    kl_from_class = p_dist.kl_divergence(q)
    kl_from_function = matrix_normal_kl(p_dist.M, p_dist.U, p_dist.V, q.M, q.U, q.V)

    assert torch.allclose(kl_from_class, kl_from_function, atol=1e-10, rtol=1e-10)


def test_kl_is_differentiable():
    """Gradients must flow through both matrix_normal_kl and matrix_normal_kl_cholesky."""
    torch.set_default_dtype(torch.float64)

    n, p = 3, 2
    M1 = torch.tensor(
        [[0.2, -0.5], [1.0, 0.3], [-0.2, 0.7]], dtype=torch.float64, requires_grad=True
    )
    M2 = torch.tensor([[-0.1, 0.4], [0.5, -0.9], [0.6, 0.2]], dtype=torch.float64)
    U1 = _spd_from_seed(n, seed=50)
    U2 = _spd_from_seed(n, seed=60)
    V1 = _spd_from_seed(p, seed=70)
    V2 = _spd_from_seed(p, seed=80)

    kl = matrix_normal_kl(M1, U1, V1, M2, U2, V2)
    kl.backward()
    assert M1.grad is not None

    M1.grad = None
    L_u1 = torch.linalg.cholesky(U1)
    L_v1 = torch.linalg.cholesky(V1)
    L_u2 = torch.linalg.cholesky(U2)
    L_v2 = torch.linalg.cholesky(V2)

    kl_chol = matrix_normal_kl_cholesky(M1, L_u1, L_v1, M2, L_u2, L_v2)
    kl_chol.backward()
    assert M1.grad is not None


def test_matrix_normal_kl_raises_for_batch_input():
    """matrix_normal_kl does not support batched inputs; it should raise a clear error."""
    torch.set_default_dtype(torch.float64)

    batch, n, p = 4, 3, 2
    M1 = torch.randn(batch, n, p, dtype=torch.float64)
    M2 = torch.randn(batch, n, p, dtype=torch.float64)
    U1 = _spd_from_seed(n, seed=50).unsqueeze(0).expand(batch, -1, -1)
    U2 = _spd_from_seed(n, seed=60).unsqueeze(0).expand(batch, -1, -1)
    V1 = _spd_from_seed(p, seed=70).unsqueeze(0).expand(batch, -1, -1)
    V2 = _spd_from_seed(p, seed=80).unsqueeze(0).expand(batch, -1, -1)

    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        matrix_normal_kl(M1, U1, V1, M2, U2, V2)
