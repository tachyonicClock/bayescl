from functools import partial
from typing import cast

import pytest
import torch
from torch import nn
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal

from bayescl.peft import BALL, BALLConfig, only_adapters_require_grad
from bayescl.peft._ball.layer import BALLLinear
from bayescl.vbnn import (
    VariationalParameter,
    VBNNConfig,
    posterior_to_prior,
)
from bayescl.vbnn import (
    kl_divergence as vbnn_kl_divergence,
)

approx = partial(pytest.approx, abs=1e-4)


def zero_grad(model: nn.Module):
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


@pytest.mark.parametrize(
    "config",
    [
        VBNNConfig(sd_mode="softplus"),
        VBNNConfig(sd_mode="abs"),
    ],
)
def test_variational_parameter(config: VBNNConfig):
    torch.manual_seed(0)
    shape = (8, 10)
    vp = VariationalParameter(shape, config)

    # ensure mu is zero-initialized
    assert torch.allclose(vp.mu, torch.zeros(shape))
    # ensure sigma (rho) is initialized correctly
    assert torch.mean(vp.sigma()).item() == pytest.approx(config.init_sd, rel=0.5)
    assert torch.std(vp.sigma()).item() == pytest.approx(config.init_sd_sd, rel=0.5)

    # ensure prior parameters are set correctly
    assert torch.all(vp.prior_mu == config.prior_mean)
    assert torch.all(vp.prior_sigma == config.prior_sd)

    # test sampling
    n_samples = 100
    samples = torch.stack([vp.forward() for _ in range(n_samples)])
    torch.testing.assert_close(samples.mean(0), vp.mu, rtol=0.1, atol=0.1)
    torch.testing.assert_close(samples.std(0), vp.sigma(), rtol=0.1, atol=0.1)

    # test kl divergence
    kl = vbnn_kl_divergence(vp)
    kl_expected = kl_divergence(
        Normal(vp.mu, vp.sigma()), Normal(vp.prior_mu, vp.prior_sigma)
    ).sum()
    assert kl.item() == approx(kl_expected.item())

    # test backwards pass
    kl.backward()
    assert vp.mu.grad is not None
    assert vp.rho.grad is not None
    # zero gradients
    vp.mu.grad.zero_()
    vp.rho.grad.zero_()

    mu = vp.mu.clone()
    sigma = vp.sigma().clone()
    posterior_to_prior(vp)
    assert torch.allclose(vp.prior_mu, mu)
    assert torch.allclose(vp.prior_sigma, sigma)

    kl = vbnn_kl_divergence(vp)
    kl_expected = kl_divergence(
        Normal(vp.mu, vp.sigma()), Normal(vp.prior_mu, vp.prior_sigma)
    ).sum()
    assert kl.item() == approx(0.0) == approx(kl_expected.item())

    kl.backward()
    assert vp.mu.grad is not None
    assert vp.rho.grad is not None
    assert torch.allclose(vp.mu.grad, torch.zeros_like(vp.mu))
    assert torch.allclose(vp.rho.grad, torch.zeros_like(vp.rho), atol=1e-6)

    vp.mu.data += 0.1
    assert vbnn_kl_divergence(vp).item() > 0.0


def test_ball():
    torch.manual_seed(0)

    in_features = 16
    out_features = 8
    batch_size = 4
    x = torch.randn(batch_size, in_features)
    model = cast(
        BALLLinear, BALL(BALLConfig(r=2))(torch.nn.Linear(in_features, out_features))
    )
    only_adapters_require_grad(model)

    # BALL is non-deterministic due to sampling
    assert not torch.allclose(model(x), model(x))

    # Ensure gradients flow correctly
    model(x).sum().backward()
    assert model.ball_A.mu.grad is not None
    assert model.ball_A.rho.grad is not None
    assert model.ball_B.mu.grad is not None
    assert model.ball_B.rho.grad is not None
    assert model.weight.grad is None
    assert model.bias.grad is None
    zero_grad(model)

    # Ensure kl divergence can be computed and backpropagated
    kl = vbnn_kl_divergence(model)
    kl.backward()
    assert model.ball_A.mu.grad is not None
    assert model.ball_A.rho.grad is not None
    assert model.ball_B.mu.grad is not None
    assert model.ball_B.rho.grad is not None
    assert model.weight.grad is None
    assert model.bias.grad is None

    # Ensure setting prior to posterior works
    mu_A = model.ball_A.mu.clone()
    sigma_A = model.ball_A.sigma().clone()
    mu_B = model.ball_B.mu.clone()
    sigma_B = model.ball_B.sigma().clone()
    posterior_to_prior(model)
    torch.testing.assert_close(model.ball_A.prior_mu, mu_A)
    torch.testing.assert_close(model.ball_A.prior_sigma, sigma_A)
    torch.testing.assert_close(model.ball_B.prior_mu, mu_B)
    torch.testing.assert_close(model.ball_B.prior_sigma, sigma_B)
    zero_grad(model)

    # Now that prior == posterior, kl divergence should be zero
    kl = vbnn_kl_divergence(model)
    assert kl.item() == approx(0.0)
    kl.backward()
    assert torch.allclose(model.ball_A.mu.grad, torch.zeros_like(model.ball_A.mu))
    assert torch.allclose(
        model.ball_A.rho.grad, torch.zeros_like(model.ball_A.rho), atol=1e-6
    )
    assert torch.allclose(model.ball_B.mu.grad, torch.zeros_like(model.ball_B.mu))
    assert torch.allclose(
        model.ball_B.rho.grad, torch.zeros_like(model.ball_B.rho), atol=1e-6
    )
