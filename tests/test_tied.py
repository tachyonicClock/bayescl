import pytest
import torch
import torch.testing as tt
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis_torch import tensor_strategy
from torch import Tensor

from bayescl.bnn.tied import (
    forward_tied_lora,
    forward_tied_lora_fast,
    kl_divergence_tied_lora,
    kl_divergence_tied_lora_fast,
)


def draw_tensor(draw: st.DrawFn, *shape: int) -> Tensor:
    return draw(
        tensor_strategy(
            dtype=torch.float64,
            shape=shape,
            layout=torch.strided,
            elements=st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
        )
    )


def draw_scale_tril(draw: st.DrawFn, dim: int) -> Tensor:
    m = draw_tensor(draw, dim, dim)
    eps = 1e-3
    # Strictly lower part from m, plus a positive diagonal
    return m.tril(-1) + torch.diag_embed(m.diagonal().abs() + eps)


@st.composite
def inputs_kl_divergence_tied_lora(
    draw: st.DrawFn,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    in_dim = draw(st.integers(min_value=1, max_value=8))
    out_dim = draw(st.integers(min_value=1, max_value=8))
    rank_dim = draw(st.integers(min_value=1, max_value=4))

    return (
        draw_tensor(draw, rank_dim, in_dim),  # A
        draw_tensor(draw, out_dim, rank_dim),  # B
        draw_scale_tril(draw, rank_dim),  # L
        draw_tensor(draw, rank_dim, in_dim),  # prior_A
        draw_tensor(draw, out_dim, rank_dim),  # prior_B
        draw_scale_tril(draw, rank_dim),  # prior_L
    )


@settings(deadline=500)
@given(inputs=inputs_kl_divergence_tied_lora())
def test_kl_divergence_tied_lora(
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
) -> None:
    kl_true = kl_divergence_tied_lora(*inputs)
    kl_fast = kl_divergence_tied_lora_fast(*inputs)

    tt.assert_close(kl_fast, kl_true, atol=1e-4, rtol=1e-6)


@st.composite
def inputs_forward_tied_lora(
    draw: st.DrawFn,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    in_dim = draw(st.integers(min_value=1, max_value=4))
    out_dim = draw(st.integers(min_value=1, max_value=4))
    rank_dim = draw(st.integers(min_value=1, max_value=2))
    batch_dim = draw(st.integers(min_value=1, max_value=2))

    return (
        draw_tensor(draw, rank_dim, in_dim),  # A
        draw_tensor(draw, out_dim, rank_dim),  # B
        draw_scale_tril(draw, rank_dim),  # L
        draw_tensor(draw, batch_dim, in_dim),  # x
    )


@settings(deadline=5000)
@given(inputs=inputs_forward_tied_lora())
def test_forward_tied_lora(inputs):
    A, B, L, x = inputs
    n_samples = 30
    n_sigma = 6  # ~1 in a billion false-positive rate for a Gaussian tail

    samples_true = torch.stack(
        [forward_tied_lora(A, B, L, x) for _ in range(n_samples)]
    )
    samples_fast = torch.stack(
        [forward_tied_lora_fast(A, B, L, x) for _ in range(n_samples)]
    )

    mean_true, mean_fast = samples_true.mean(0), samples_fast.mean(0)
    var_true, var_fast = samples_true.var(0), samples_fast.var(0)

    # Use the pooled/larger of the two empirical std devs as a stand-in for sigma
    std_est = torch.maximum(samples_true.std(0), samples_fast.std(0))

    mean_atol = n_sigma * std_est * (2 / n_samples) ** 0.5
    var_atol = (
        n_sigma * (std_est**2) * (2 * 2 / (n_samples - 1)) ** 0.5
    )  # sqrt(2) extra factor for two independent estimates

    tt.assert_close(mean_true, mean_fast, atol=mean_atol.max().item(), rtol=0.0)
    tt.assert_close(var_true, var_fast, atol=var_atol.max().item(), rtol=0.0)


@pytest.fixture(scope="module")
def benchmark_inputs():
    torch.manual_seed(0)
    rank_dim = 8
    in_dim, out_dim = 768, 768
    dtype = torch.float
    A = torch.randn(rank_dim, in_dim, dtype=dtype)
    B = torch.randn(out_dim, rank_dim, dtype=dtype)
    prior_A = torch.zeros(rank_dim, in_dim, dtype=dtype)
    prior_B = torch.zeros(out_dim, rank_dim, dtype=dtype)

    # Cholesky factors of well-conditioned random PD matrices for L, prior_L
    M = torch.randn(rank_dim, rank_dim, dtype=dtype)
    L = torch.linalg.cholesky(M @ M.T + rank_dim * torch.eye(rank_dim, dtype=dtype))
    prior_L = torch.eye(rank_dim, dtype=dtype)

    return A, B, L, prior_A, prior_B, prior_L


def test_benchmark_kl_divergence_tied_lora(benchmark, benchmark_inputs):
    A, B, L, prior_A, prior_B, prior_L = benchmark_inputs
    benchmark(kl_divergence_tied_lora, A, B, L, prior_A, prior_B, prior_L)


def test_benchmark_kl_divergence_tied_lora_fast(benchmark, benchmark_inputs):
    A, B, L, prior_A, prior_B, prior_L = benchmark_inputs
    benchmark(kl_divergence_tied_lora_fast, A, B, L, prior_A, prior_B, prior_L)


@pytest.fixture(scope="module")
def benchmark_forward_inputs():
    torch.manual_seed(0)
    rank_dim = 8
    in_dim, out_dim = 768, 768
    batch_dim = 32
    dtype = torch.float
    A = torch.randn(rank_dim, in_dim, dtype=dtype)
    B = torch.randn(out_dim, rank_dim, dtype=dtype)
    x = torch.randn(batch_dim, in_dim, dtype=dtype)

    # Cholesky factor of a well-conditioned random PD matrix for L
    M = torch.randn(rank_dim, rank_dim, dtype=dtype)
    L = torch.linalg.cholesky(M @ M.T + rank_dim * torch.eye(rank_dim, dtype=dtype))

    return A, B, L, x


def test_benchmark_forward_tied_lora(benchmark, benchmark_forward_inputs):
    A, B, L, x = benchmark_forward_inputs
    benchmark(forward_tied_lora, A, B, L, x)


def test_benchmark_forward_tied_lora_fast(benchmark, benchmark_forward_inputs):
    A, B, L, x = benchmark_forward_inputs
    benchmark(forward_tied_lora_fast, A, B, L, x)
