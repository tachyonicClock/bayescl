from typing import Tuple

import torch.distributions as dist
from avalanche.training.supervised import Naive
from bnn.nn.mixins.variational.base import VariationalMixin
from loguru import logger
from torch import BoolTensor, Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import cross_entropy, nll_loss

from bayescl.peft._ball.layer import posterior_to_prior


class BALLStrategy(Naive):
    def __init__(
        self,
        beta: float,
        train_samples: int,
        test_samples: int,
        mask: BoolTensor,
        train_data_size: int,
        writer: SummaryWriter | None = None,
        **kwargs,
    ):
        assert "criterion" not in kwargs, "criterion is set by BALL"
        super().__init__(**kwargs)
        if train_samples < 1:
            raise ValueError("train_samples must be at least 1")
        if test_samples < 1:
            raise ValueError("test_samples must be at least 1")

        self.beta = beta
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.train_data_size = train_data_size
        self.writer = writer
        self.mask = mask.to(self.device)

        logger.info(
            "Initialized `BALLStrategy` with"
            f" beta={self.beta} train_samples={self.train_samples}"
            f" test_samples={self.test_samples} train_data_size={self.train_data_size}"
        )

    def kl_loss(self) -> Tensor:
        return sum(
            m.kl_divergence()
            for m in self.model.modules()
            if isinstance(m, VariationalMixin)
        )  # type: ignore

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        assert not self.is_eval

        t = self.clock.train_exp_counter
        x, y, _ = batch

        kl = self.kl_loss()
        pred_probs = 0
        nll = 0
        for k in range(self.train_samples):
            y_hat = self.mask[t] * self.model(x)
            pred_probs += y_hat.softmax(dim=-1)
            nll += cross_entropy(y_hat, y)

        # Average over samples
        pred_probs /= self.train_samples
        nll /= self.train_samples
        kl /= self.train_data_size

        beta_kl = self.beta * kl
        loss = nll + beta_kl

        self.writer.add_scalar("ball/nll", nll.item(), self.clock.train_iterations)
        self.writer.add_scalar("ball/kl", kl.item(), self.clock.train_iterations)
        self.writer.add_scalar(
            "ball/beta_kl", beta_kl.item(), self.clock.train_iterations
        )
        return loss, pred_probs

    def predict_step(
        self, batch: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        assert self.is_eval

        n = self.test_samples
        x, y, _ = batch
        # Bayesian Posterior Predictive Distribution (marginalize over the model posterior)
        # Like ensembling, but each ensemble member is a sample from the model posterior.
        pred_probs: Tensor = sum(self.model(x).softmax(dim=-1) for _ in range(n)) / n # type: ignore
        loss = nll_loss(pred_probs.log(), y)
        return loss, pred_probs

    def training_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()

            # Forward
            self._before_forward(**kwargs)
            loss, mb_output = self.training_step(self.mbatch)
            self.mb_output = mb_output
            self.loss += loss
            self._after_forward(**kwargs)

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)
            self.loss, self.mb_output = self.predict_step(self.mbatch)
            self._after_eval_forward(**kwargs)

            self._after_eval_iteration(**kwargs)

    def _after_training_exp(self, **kwargs):
        logger.info("Using previous posterior as new prior.")
        posterior_to_prior(self.model)
        return super()._after_training_exp(**kwargs)
