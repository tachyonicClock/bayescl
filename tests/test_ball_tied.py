"""Tests for the tied-covariance BALL adapter.

The headline property is identifiability: mean-field BALL admits a rescaling
``A -> cA, B -> B/c`` that leaves the predictive distribution untouched while
moving the KL, and tying the covariance is supposed to remove it. Those two
facts are asserted side by side below so the contrast is what's under test, not
just the tied arm in isolation.
"""

import math

import pytest
import torch
from torch import nn

from bayescl.bnn.tied import (
    forward_tied_lora_fast,
    kl_divergence_tied_lora_fast,
    sample_tied_lora_weights,
)
from bayescl.bnn.vbnn import posterior_to_prior
from bayescl.treatments.ball._tied_config import TiedBALLConfig
from bayescl.treatments.ball._tied_module import (
    TiedBALLAdapterFactory,
    TiedBALLLinear,
    TiedLoRAParameter,
)


def balanced_scale(in_features: int, rank_dim: int) -> float:
    """The width the shared prior is scaled by, mirrored from the module.

    Each factor's own fan-in scale differs -- ``1/sqrt(in_features)`` for ``A``,
    ``1/sqrt(rank_dim)`` for ``B`` -- and one shared covariance cannot hold
    both. It doesn't need to: ``B @ A`` is invariant to ``A -> kA, B -> B/k``,
    and the two coincide at ``k = (in_features / rank_dim) ** 0.25`` where both
    become ``(in_features * rank_dim) ** -0.25``.
    """
    return (in_features * rank_dim) ** 0.25

RANK, IN_DIM, OUT_DIM, BATCH = 4, 16, 8, 8192
SAMPLES = 400


@pytest.fixture
def tied():
    torch.manual_seed(0)
    A = torch.randn(RANK, IN_DIM) / math.sqrt(IN_DIM)
    B = torch.randn(OUT_DIM, RANK) / math.sqrt(RANK)
    L = torch.linalg.cholesky(torch.eye(RANK) * 0.04 + 0.01 * torch.ones(RANK, RANK))
    x = torch.randn(BATCH, IN_DIM)
    prior_A = torch.zeros(RANK, IN_DIM)
    prior_B = torch.zeros(OUT_DIM, RANK)
    prior_L = torch.eye(RANK) * 0.5
    return A, B, L, x, prior_A, prior_B, prior_L


def _predictive_var(forward) -> torch.Tensor:
    return torch.stack([forward() for _ in range(SAMPLES)]).var(0).mean()


class TestGaugeFreedom:
    """``A -> cA, B -> B/c`` must stop being a flat direction of the ELBO."""

    @pytest.mark.parametrize("c", [0.25, 4.0])
    def test_mean_field_rescaling_is_a_flat_direction(self, c):
        """Baseline: the defect the tied parameterization is meant to remove.

        Mean-field BALL's predictive distribution is *exactly* invariant to the
        rescaling, so the KL alone decides ``c``.
        """
        torch.manual_seed(0)
        mu_A = torch.randn(RANK, IN_DIM) / math.sqrt(IN_DIM)
        mu_B = torch.randn(OUT_DIM, RANK) / math.sqrt(RANK)
        sd_A = torch.full((RANK, IN_DIM), 0.2)
        sd_B = torch.full((OUT_DIM, RANK), 0.2)
        x = torch.randn(BATCH, IN_DIM)

        def forward(mu_A, sd_A, mu_B, sd_B):
            torch.manual_seed(1234)  # same noise draw, so equality is exact
            A = mu_A + sd_A * torch.randn(BATCH, RANK, IN_DIM)
            B = mu_B + sd_B * torch.randn(BATCH, OUT_DIM, RANK)
            return (B @ (A @ x.unsqueeze(-1))).squeeze(-1)

        def kl(mu, sd, prior_sd=0.5):
            p = torch.full_like(sd, prior_sd)
            return (p.log() - sd.log() + (sd**2 + mu**2) / (2 * p**2) - 0.5).sum()

        base = forward(mu_A, sd_A, mu_B, sd_B)
        scaled = forward(mu_A * c, sd_A * c, mu_B / c, sd_B / c)
        torch.testing.assert_close(base, scaled, atol=1e-4, rtol=1e-4)

        kl_base = kl(mu_A, sd_A) + kl(mu_B, sd_B)
        kl_scaled = kl(mu_A * c, sd_A * c) + kl(mu_B / c, sd_B / c)
        assert not torch.isclose(kl_base, kl_scaled, rtol=0.05)

    @pytest.mark.parametrize("c", [0.25, 4.0])
    def test_tied_rescaling_changes_the_predictive_distribution(self, tied, c):
        """The same rescaling is no longer absorbed by the tied covariance.

        ``L`` is scaled by ``c`` too, which is the best the family can do: it
        matches the ``A``-path noise, but then the ``B``-path noise picks up a
        factor of ``c**2``. So the direction moves the likelihood and the KL
        term is no longer free to choose ``c`` on its own.
        """
        A, B, L, x, *_ = tied
        var_base = _predictive_var(lambda: forward_tied_lora_fast(A, B, L, x))
        var_scaled = _predictive_var(
            lambda: forward_tied_lora_fast(A * c, B / c, L * c, x)
        )
        assert not torch.isclose(var_base, var_scaled, rtol=0.1)

    def test_tied_is_invariant_to_rotations_of_the_rank_space(self, tied):
        """The residual ``O(r)`` symmetry is benign: it moves neither term.

        With a zero-mean isotropic prior, ``A -> QA, B -> BQ^T`` leaves the
        predictive distribution *and* the KL fixed, so it is a relabelling of
        the rank axes rather than a direction the KL can exploit.
        """
        A, B, L, x, prior_A, prior_B, prior_L = tied
        var_base = _predictive_var(lambda: forward_tied_lora_fast(A, B, L, x))
        kl_base = kl_divergence_tied_lora_fast(A, B, L, prior_A, prior_B, prior_L)

        for _ in range(3):
            Q = torch.linalg.qr(torch.randn(RANK, RANK))[0]
            A_q, B_q = Q @ A, B @ Q.T
            L_q = torch.linalg.cholesky(Q @ (L @ L.T) @ Q.T)

            var_q = _predictive_var(lambda: forward_tied_lora_fast(A_q, B_q, L_q, x))
            kl_q = kl_divergence_tied_lora_fast(
                A_q, B_q, L_q, prior_A, prior_B, prior_L
            )

            torch.testing.assert_close(var_q, var_base, rtol=0.05, atol=0.0)
            torch.testing.assert_close(kl_q, kl_base, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("c", [0.25, 4.0])
    def test_weight_sampling_also_breaks_the_rescaling(self, tied, c):
        """The fix must come from the family, not the sampling scheme.

        ``sampling="weight"`` shares one draw across the batch instead of
        drawing per row, so if the rescaling were flat here the cheap mode
        would quietly give up the identifiability the tied posterior buys.
        """
        A, B, L, x, *_ = tied

        def predictive_var(A_, B_, L_):
            draws = []
            for _ in range(SAMPLES):
                A_s, B_s = sample_tied_lora_weights(A_, B_, L_)
                draws.append((x @ A_s.T) @ B_s.T)
            return torch.stack(draws).var(0).mean()

        var_base = predictive_var(A, B, L)
        var_scaled = predictive_var(A * c, B / c, L * c)
        assert not torch.isclose(var_base, var_scaled, rtol=0.1)

    def test_rotation_symmetry_breaks_once_the_prior_is_a_learned_posterior(self, tied):
        """After a task boundary the prior mean is non-zero, breaking ``O(r)`` too."""
        A, B, L, _, _, _, prior_L = tied
        prior_A = torch.randn(RANK, IN_DIM) / math.sqrt(IN_DIM)
        prior_B = torch.randn(OUT_DIM, RANK) / math.sqrt(RANK)

        kl_base = kl_divergence_tied_lora_fast(A, B, L, prior_A, prior_B, prior_L)
        Q = torch.linalg.qr(torch.randn(RANK, RANK))[0]
        kl_q = kl_divergence_tied_lora_fast(
            Q @ A,
            B @ Q.T,
            torch.linalg.cholesky(Q @ (L @ L.T) @ Q.T),
            prior_A,
            prior_B,
            prior_L,
        )
        assert not torch.isclose(kl_base, kl_q, rtol=1e-3)


class TestSampleTiedLoRAWeights:
    def test_draws_match_the_shared_covariance(self):
        """Columns of A and rows of B must both be drawn from S = L L^T."""
        torch.manual_seed(0)
        A = torch.zeros(RANK, IN_DIM)
        B = torch.zeros(OUT_DIM, RANK)
        L = torch.linalg.cholesky(torch.eye(RANK) * 0.4 + 0.1 * torch.ones(RANK, RANK))
        S = L @ L.T

        n = 20000
        draws = [sample_tied_lora_weights(A, B, L) for _ in range(n)]
        # Column 0 of A and row 0 of B, stacked over draws
        a_cols = torch.stack([d[0][:, 0] for d in draws])
        b_rows = torch.stack([d[1][0, :] for d in draws])

        tol = 4 * S.abs().max().item() / math.sqrt(n)
        for label, sample in (("A columns", a_cols), ("B rows", b_rows)):
            cov = (sample - sample.mean(0)).T @ (sample - sample.mean(0)) / (n - 1)
            torch.testing.assert_close(cov, S, atol=tol, rtol=0.0, msg=label)

    def test_is_differentiable_through_the_means_and_factor(self):
        A = torch.zeros(RANK, IN_DIM, requires_grad=True)
        B = torch.zeros(OUT_DIM, RANK, requires_grad=True)
        L = (torch.eye(RANK) * 0.1).requires_grad_(True)
        A_s, B_s = sample_tied_lora_weights(A, B, L)
        (A_s.sum() + B_s.sum()).backward()
        for p in (A, B, L):
            assert p.grad is not None and torch.isfinite(p.grad).all()


class TestTiedLoRAParameter:
    def test_kl_is_zero_when_the_posterior_equals_the_prior(self):
        # ``prior_sd`` is a fan-in scaled width while ``init_sd`` is an absolute
        # entry scale, so lining the two up takes the scale factor. The two
        # previously coincided only because the prior went unscaled.
        prior_sd = 0.1
        config = TiedBALLConfig(
            r=RANK,
            prior_sd=prior_sd,
            init_sd=prior_sd / balanced_scale(IN_DIM, RANK),
        )
        param = TiedLoRAParameter(IN_DIM, OUT_DIM, config)
        with torch.no_grad():
            param.A.copy_(param.prior_A)
            param.B.copy_(param.prior_B)
        torch.testing.assert_close(param.L, param.prior_L, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(
            param.kl_divergence(), torch.tensor(0.0), atol=1e-3, rtol=0.0
        )

    def test_prior_width_is_fan_in_scaled(self):
        """The shared prior is scaled by layer width, as the mean-field
        adapter's is -- an absolute ``prior_sd`` would sit orders of magnitude
        above the factors it regularizes once ``in_features`` grows, and would
        make the two arms differ in prior as well as posterior family."""
        config = TiedBALLConfig(r=RANK, prior_sd=1.0)
        param = TiedLoRAParameter(IN_DIM, OUT_DIM, config)
        expected = 1.0 / balanced_scale(IN_DIM, RANK)
        torch.testing.assert_close(
            param.prior_L, torch.eye(RANK) * expected, atol=1e-6, rtol=0.0
        )
        assert expected < 1.0

    def test_balanced_scale_equalizes_both_factors_fan_ins(self):
        """``A``'s and ``B``'s own fan-in scales differ, but ``B @ A`` is
        invariant to ``A -> kA, B -> B/k``; the balanced gauge is where the two
        coincide, and that common value is what one shared covariance can hold.
        """
        in_features, rank = 384, 8
        k = (in_features / rank) ** 0.25
        a_scale = k / math.sqrt(in_features)  # A's fan-in scale, rescaled by k
        b_scale = 1.0 / (k * math.sqrt(rank))  # B's, rescaled by 1/k
        assert a_scale == pytest.approx(b_scale)
        assert a_scale == pytest.approx(1.0 / balanced_scale(in_features, rank))

    @pytest.mark.parametrize("init_sd", [1e-3, 0.1])
    def test_l_starts_isotropic_at_init_sd(self, init_sd):
        param = TiedLoRAParameter(
            IN_DIM, OUT_DIM, TiedBALLConfig(r=RANK, init_sd=init_sd)
        )
        L = param.L
        torch.testing.assert_close(
            L, torch.eye(RANK) * init_sd, atol=init_sd * 1e-3, rtol=0.0
        )

    @pytest.mark.parametrize("init_sd", [1e-3, 0.1])
    def test_conditioning_does_not_depend_on_init_sd(self, init_sd):
        """Off-diagonal drift must not be able to make ``L`` near-singular.

        Gradient steps displace ``L_raw``'s off-diagonals by an absolute amount
        set by the learning rate, not one that shrinks with ``init_sd``. Held
        absolutely, a few steps' worth would swamp a small diagonal and leave a
        near-singular ``L`` that the next task inherits as ``prior_L``.
        """
        param = TiedLoRAParameter(
            IN_DIM, OUT_DIM, TiedBALLConfig(r=RANK, init_sd=init_sd)
        )
        with torch.no_grad():
            param.L_raw.add_(torch.full((RANK, RANK), 3e-3).tril(-1))
        assert torch.linalg.cond(param.L).item() < 1.5

    def test_kl_is_positive_and_differentiable(self):
        param = TiedLoRAParameter(IN_DIM, OUT_DIM, TiedBALLConfig(r=RANK))
        kl = param.kl_divergence()
        assert kl.item() > 0
        kl.backward()
        for p in (param.A, param.B, param.L_raw):
            assert p.grad is not None and torch.isfinite(p.grad).all()

    def test_posterior_to_prior_zeroes_the_kl(self):
        param = TiedLoRAParameter(IN_DIM, OUT_DIM, TiedBALLConfig(r=RANK))
        assert param.kl_divergence().item() > 0
        posterior_to_prior(param)
        torch.testing.assert_close(
            param.kl_divergence(), torch.tensor(0.0), atol=1e-4, rtol=0.0
        )

    @pytest.mark.parametrize("sampling", ["weight", "lrt"])
    @pytest.mark.parametrize("shape", [(5, IN_DIM), (3, 7, IN_DIM)])
    def test_forward_preserves_leading_dimensions(self, shape, sampling):
        param = TiedLoRAParameter(
            IN_DIM, OUT_DIM, TiedBALLConfig(r=RANK, sampling=sampling)
        )
        out = param(torch.randn(*shape))
        assert out.shape == (*shape[:-1], OUT_DIM)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("sampling", ["weight", "lrt"])
    def test_forward_returns_input_dtype_under_bf16_autocast(self, sampling):
        param = TiedLoRAParameter(
            IN_DIM, OUT_DIM, TiedBALLConfig(r=RANK, sampling=sampling)
        )
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = param(torch.randn(5, IN_DIM, dtype=torch.bfloat16))
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()

    def test_weight_sampling_shares_one_draw_across_the_batch(self):
        """Identical rows in must give identical rows out under ``weight``.

        This is what makes the mode cheap, and it is the property that
        distinguishes it from ``lrt``, where every row is drawn independently.
        """
        row = torch.randn(1, IN_DIM)
        x = row.repeat(16, 1)

        shared = TiedLoRAParameter(
            IN_DIM, OUT_DIM, TiedBALLConfig(r=RANK, sampling="weight")
        )(x)
        torch.testing.assert_close(shared, shared[:1].expand_as(shared))

        per_row = TiedLoRAParameter(
            IN_DIM, OUT_DIM, TiedBALLConfig(r=RANK, sampling="lrt")
        )(x)
        assert not torch.allclose(per_row, per_row[:1].expand_as(per_row))


class TestTiedBALLLinear:
    @pytest.mark.parametrize("sampling", ["weight", "lrt"])
    def test_starts_as_the_wrapped_linear_in_expectation(self, sampling):
        torch.manual_seed(0)
        base = nn.Linear(IN_DIM, OUT_DIM)
        adapter = TiedBALLAdapterFactory(
            TiedBALLConfig(r=RANK, init_sd=1e-6, sampling=sampling)
        )(base)
        assert isinstance(adapter, TiedBALLLinear)

        x = torch.randn(32, IN_DIM)
        torch.testing.assert_close(adapter(x), base(x), atol=1e-4, rtol=1e-4)

    def test_only_adapter_parameters_are_exposed_as_tunable(self):
        adapter = TiedBALLLinear(IN_DIM, OUT_DIM, TiedBALLConfig(r=RANK))
        names = set(adapter.adapter_parameters)
        assert names == {"ball.A", "ball.B", "ball.L_raw"}
        for name in names:
            assert adapter.get_parameter(name) is not None

    def test_rejects_unsupported_layer_types(self):
        factory = TiedBALLAdapterFactory(TiedBALLConfig(r=RANK))
        with pytest.raises(ValueError, match="Unsupported layer type"):
            factory(nn.Conv2d(3, 3, 3))
