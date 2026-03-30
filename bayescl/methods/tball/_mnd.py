from typing import Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from bayescl.mnd.mnd import matrix_normal_kl_cholesky, matrix_normal_sample_cholesky
from bayescl.vbnn import MatrixNormalPriorPosterior, inv_softplus

from ._config import TBALLConfig


class MNDParameter(MatrixNormalPriorPosterior):
    def __init__(
        self,
        shape: Tuple[int, int],
        config: TBALLConfig,
    ):
        super().__init__()
        n, p = shape
        self.M = nn.Parameter(torch.zeros(n, p))
        self.L_u_raw = nn.Parameter(torch.zeros(n, n))
        self.L_v_raw = nn.Parameter(torch.zeros(p, p))

        init = inv_softplus(torch.sqrt(torch.tensor(config.init_sd)))
        nn.init.constant_(self.L_u_raw.diagonal(), init.item())
        nn.init.constant_(self.L_v_raw.diagonal(), init.item())

        self.register_buffer("prior_M", torch.full((n, p), config.prior_mean))
        prior_cov = torch.tensor(config.prior_weight_sd)
        self.register_buffer(
            "prior_L_u", torch.linalg.cholesky(torch.eye(n) * prior_cov)
        )
        self.register_buffer(
            "prior_L_v", torch.linalg.cholesky(torch.eye(p) * prior_cov)
        )

    @staticmethod
    def _get_valid_cholesky(L_raw: Tensor) -> Tensor:
        L = torch.tril(L_raw, diagonal=-1)
        diag = F.softplus(L_raw.diagonal()) + 1e-4
        return L + torch.diag_embed(diag)

    @property
    def L_u(self) -> Tensor:
        return self._get_valid_cholesky(self.L_u_raw)

    @property
    def L_v(self) -> Tensor:
        return self._get_valid_cholesky(self.L_v_raw)

    @property
    def U(self) -> Tensor:
        return self.L_u @ self.L_u.T

    @property
    def V(self) -> Tensor:
        return self.L_v @ self.L_v.T

    def sample(self) -> Tensor:
        return matrix_normal_sample_cholesky(self.M, self.L_u, self.L_v)

    def kl_divergence(self) -> Tensor:
        prior_M = cast(Tensor, self.prior_M)
        prior_L_u = cast(Tensor, self.prior_L_u)
        prior_L_v = cast(Tensor, self.prior_L_v)
        return matrix_normal_kl_cholesky(
            self.M,
            self.L_u,
            self.L_v,
            prior_M,
            prior_L_u,
            prior_L_v,
        )


class MNDLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, config: TBALLConfig):
        super().__init__()
        if config.bias:
            raise NotImplementedError("MND bayes_core does not support bias yet")

        self.weight = MNDParameter((out_features, in_features), config)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.sample(), None)
