import math
from typing import Any, Callable, Tuple, override

import torch
from avalanche.training.supervised import Naive
from loguru import logger
from torch import BoolTensor, Tensor
from torch.nn.functional import nll_loss
from torch.utils.tensorboard import SummaryWriter

from bayescl.base import NumericError
from bayescl.vbnn import (
    kl_divergence,
    posterior_to_prior,
)

torch.autograd.set_detect_anomaly(True)


class VCLStrategy(Naive):
    def __init__(
        self,
        *,
        beta: float,
        train_samples: int,
        test_samples: int,
        mask: BoolTensor,
        optimizer_fn: Callable[[Any], torch.optim.Optimizer],
        writer: SummaryWriter,
        softmax_avg: bool,
        **kwargs,
    ):
        del kwargs["criterion"]  # VCL uses its own criterion
        super().__init__(**kwargs)
        if train_samples < 1:
            raise ValueError("train_samples must be at least 1")
        if test_samples < 1:
            raise ValueError("test_samples must be at least 1")

        self._optimizer_fn = optimizer_fn
        self._beta = beta
        self._softmax_avg = softmax_avg
        self._train_samples = train_samples
        self._test_samples = test_samples
        self._writer = writer
        self._mask = mask.to(self.device)

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        assert not self.is_eval
        mask = self._mask[self.clock.train_exp_counter]
        x, y, _ = batch

        ys_hat = torch.stack([mask * self.model(x) for _ in range(self._train_samples)])
        if self._softmax_avg:
            log_pred_probs = ys_hat.log_softmax(dim=-1).logsumexp(dim=0)
            log_pred_probs = log_pred_probs - math.log(self._train_samples)
        else:
            log_pred_probs = ys_hat.mean(dim=0).log_softmax(dim=-1)

        nll: Tensor = nll_loss(log_pred_probs, y)
        kl: Tensor = kl_divergence(self.model) / len(self.experience.dataset)  # type: ignore

        beta_kl = self._beta * kl
        if not torch.isfinite(beta_kl):
            raise NumericError(
                f"KL divergence is NaN or Inf: {kl.item()} (beta={self._beta})"
            )
        if not torch.isfinite(nll):
            raise NumericError(f"NLL loss is NaN or Inf: {nll.item()}")

        loss = nll + beta_kl

        step = self.clock.train_iterations
        self._writer.add_scalar("vcl/nll", nll, step)
        self._writer.add_scalar("vcl/kl", kl, step)
        self._writer.add_scalar("vcl/beta_kl", beta_kl, step)

        return log_pred_probs.exp(), loss

        # pred_probs = 0
        # nll = 0
        # for k in range(self.train_samples):
        #     y_hat = self.mask[t] * self.model(x)
        #     pred_probs += y_hat.softmax(dim=-1)
        #     nll += cross_entropy(y_hat, y)

        # pred_probs /= self.train_samples
        # nll /= self.train_samples

        # # Scale the KL divergence by the number of samples in the dataset.
        # # The idea is that the cross-entropy loss is an average over a mini-batch and
        # # is therefore 1/dataset_size to small compared to the likelihood
        # kl = kl_divergence(self.model) / len(self.experience.dataset)  # type: ignore
        # beta_kl = self.beta * kl
        # loss = nll + beta_kl

        # step = self.clock.train_iterations
        # self.writer.add_scalar("ball/nll", nll, step)
        # self.writer.add_scalar("ball/kl", kl, step)
        # self.writer.add_scalar("ball/beta_kl", beta_kl, step)
        # return pred_probs, loss # type: ignore

    # def _after_backward(self, **kwargs):
    #     # Gradient clipping for variational parameters
    #     torch.nn.utils.clip_grad_norm_(
    #         [p for p in self.model.parameters() if p.requires_grad],
    #         max_norm=1.0,
    #     )
    #     return super()._after_backward(**kwargs)

    def predict_step(
        self, batch: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        x, y, _ = batch
        # Bayesian Posterior Predictive Distribution (marginalize over the model posterior)
        # Like ensembling, but each ensemble member is a sample from the model posterior.
        ys_hat = torch.stack([self.model(x) for _ in range(self._test_samples)])
        if self._softmax_avg:
            # Apply softmax to each sample and then average
            pred_probs = ys_hat.softmax(dim=-1).mean(dim=0)
        else:
            # Average the logits and then apply softmax
            pred_probs = ys_hat.mean(dim=0).softmax(dim=-1)
        return pred_probs, nll_loss(pred_probs.log(), y)

    @override
    def _before_training_exp(self, **kwargs):
        # Reset the momentum optimizer to avoid forgetting previous tasks because of
        # stale momentum.
        self.optimizer = self._optimizer_fn(self.model.parameters())  # type: ignore
        return super()._before_training_exp(**kwargs)

    @override
    def training_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()

            # Forward
            self._before_forward(**kwargs)
            self.mb_output, self.loss = self.training_step(self.mbatch)  # type: ignore
            self._after_forward(**kwargs)

            # Backward
            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    @override
    def eval_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)
            self.mb_output, self.loss = self.predict_step(self.mbatch)  # type: ignore
            self._after_eval_forward(**kwargs)

            self._after_eval_iteration(**kwargs)

    @override
    def _after_training_exp(self, **kwargs):
        logger.info("Using previous posterior as new prior.")
        posterior_to_prior(self.model)  # type: ignore
        return super()._after_training_exp(**kwargs)
