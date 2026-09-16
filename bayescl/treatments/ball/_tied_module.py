"""Tied-covariance variant of the BALL adapter.

Mean-field BALL puts an independent Gaussian on every entry of ``A`` and ``B``.
That family is closed under ``A -> cA, B -> B/c``, which leaves the predictive
distribution exactly unchanged while moving the KL term, so the ELBO has a flat
likelihood direction that only the prior arbitrates. Here ``A``'s columns and
``B``'s rows instead share one ``r x r`` covariance ``S = L L^T``, which the
rescaling cannot be absorbed into, so that direction is no longer flat.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from bayescl.bnn.tied import (
    forward_tied_lora_fast,
    kl_divergence_tied_lora_fast,
    sample_tied_lora_weights,
)
from bayescl.bnn.vbnn import TiedLoRAPriorPosterior, inv_softplus
from bayescl.peft._base import AdapterBase, AdapterFactory

from ._tied_config import TiedBALLConfig

#: Floor on the diagonal of ``L`` so it stays a strictly positive-definite
#: Cholesky factor, and so ``prior_L`` stays invertible once a finished
#: posterior is copied into it at a task boundary.
_DIAG_EPS = 1e-4


class TiedLoRAParameter(TiedLoRAPriorPosterior):
    """Variational parameters for one tied Bayesian LoRA pair."""

    def __init__(self, in_features: int, out_features: int, config: TiedBALLConfig):
        super().__init__()
        r = config.r
        self.sampling = config.sampling
        self.A = nn.Parameter(torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.L_raw = nn.Parameter(torch.zeros(r, r))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B's mean stays at zero so the adapter contributes no mean shift at the
        # start of task 0, as in standard LoRA.
        init = inv_softplus(torch.tensor(config.init_sd - _DIAG_EPS).clamp(min=1e-6))
        nn.init.constant_(self.L_raw.diagonal(), init.item())

        self.register_buffer("prior_A", torch.full((r, in_features), config.prior_mean))
        self.register_buffer(
            "prior_B", torch.full((out_features, r), config.prior_mean)
        )
        self.register_buffer("prior_L", torch.eye(r) * config.prior_sd)

    @property
    def L(self) -> Tensor:
        """Lower-triangular Cholesky factor of the shared covariance.

        Built as ``diag(d) @ (I + strictly_lower(L_raw))`` rather than
        ``strictly_lower(L_raw) + diag(d)``, so each row's off-diagonal entries
        are expressed *relative* to that row's scale. Stored absolutely, the
        off-diagonals drift at the optimizer's step size no matter how small
        ``d`` is, and for a small ``init_sd`` they overtake the diagonal within
        a few steps; ``L`` then becomes near-singular, and the next task
        inherits it as ``prior_L`` and sees the KL explode. Row-scaling makes
        the conditioning independent of ``init_sd``.
        """
        d = F.softplus(self.L_raw.diagonal()) + _DIAG_EPS
        return self.L_raw.tril(-1) * d.unsqueeze(-1) + torch.diag_embed(d)

    def kl_divergence(self) -> Tensor:
        return kl_divergence_tied_lora_fast(
            self.A, self.B, self.L, self.prior_A, self.prior_B, self.prior_L
        )

    def forward(self, x: Tensor) -> Tensor:
        """Sample ``B @ A @ x`` for inputs of any leading shape."""
        if self.sampling == "weight":
            # One draw shared by the whole batch. The matmuls are left to
            # autocast, matching how the mean-field adapter runs.
            A, B = sample_tied_lora_weights(self.A, self.B, self.L)
            return (x @ A.T) @ B.T

        # Local reparameterization: the tied forward is defined for a 2D batch,
        # so leading dimensions (e.g. a transformer's sequence axis) are folded
        # into the batch and each resulting row draws its own weights. Inputs
        # are cast to the parameter dtype so sampling stays in fp32 under bf16
        # autocast.
        dtype = self.A.dtype
        flat = x.reshape(-1, x.shape[-1]).to(dtype)
        out = forward_tied_lora_fast(self.A, self.B, self.L, flat)
        return out.reshape(*x.shape[:-1], -1).to(x.dtype)


class TiedBALLLinear(nn.Linear, AdapterBase):
    """A Bayesian adaptation layer using tied BALL for Linear layers."""

    adapter_parameters = ("ball.A", "ball.B", "ball.L_raw")

    def __init__(self, in_features, out_features, config: TiedBALLConfig, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        AdapterBase.__init__(self)
        self.ball = TiedLoRAParameter(in_features, out_features, config)
        self.scaling = config.lora_alpha / config.r
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

    def forward(self, input: Tensor) -> Tensor:
        z = self.ball(self.dropout(input))
        return F.linear(input, self.weight, self.bias) + z * self.scaling


class TiedBALLAdapterFactory(AdapterFactory):
    """A factory for creating tied BALL adapters."""

    def __init__(self, config: TiedBALLConfig) -> None:
        self.config = config

    def _get_replacement(self, module: nn.Module) -> nn.Module:
        if isinstance(module, nn.Linear):
            return TiedBALLLinear(
                module.in_features, module.out_features, config=self.config
            )
        raise ValueError(f"Unsupported layer type: {type(module)}")

    def __call__(self, module: nn.Module) -> nn.Module:
        """Create an adapter for a given module."""
        replacement = self._get_replacement(module)
        replacement.load_state_dict(module.state_dict(), strict=False)
        return replacement
