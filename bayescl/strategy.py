from typing import Callable, Tuple

import torch
from avalanche.training.supervised import Naive
from loguru import logger
from torch import BoolTensor, Tensor, nn
from torch.nn.functional import cross_entropy, nll_loss
from torch.utils.tensorboard import SummaryWriter

from bayescl.batch_ensemble import ensemble_predict
from bayescl.config import BALL
from bayescl.vbnn import (
    VariationalLinear,
    VariationalParameter,
    kl_divergence,
    posterior_to_prior,
)


class BALLStrategy(Naive):
    def __init__(
        self,
        *,
        mask: BoolTensor,
        optimizer_fn: Callable[[], torch.optim.Optimizer],
        writer: SummaryWriter,
        config: BALL,
        **kwargs,
    ):
        assert "criterion" not in kwargs, "criterion is set by BALL"
        super().__init__(**kwargs)
        self.cfg = config
        if self.cfg.train_samples < 1:
            raise ValueError("train_samples must be at least 1")
        if self.cfg.test_samples < 1:
            raise ValueError("test_samples must be at least 1")

        self.optimizer_fn = optimizer_fn
        self.writer = writer
        self.mask = mask.to(self.device)

        logger.info("Initialized `BALLStrategy`")

    @property
    def beta(self) -> float:
        # Use a different beta for the first task if specified
        epoch = self.clock.train_exp_epochs
        if self.cfg.beta_anneal_epochs <= 0:
            return self.cfg.beta
        else:
            return min(
                self.cfg.beta, self.cfg.beta * (epoch + 1) / self.cfg.beta_anneal_epochs
            )

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        assert not self.is_eval
        encoder: nn.Module = self.model.model.vit  # type: ignore
        head: nn.Module = self.model.model.classifier  # type: ignore

        t = self.clock.train_exp_counter
        x, y, _ = batch

        # Monte Carlo estimate of the expected log-likelihood
        pred_probs = 0
        nll = 0
        for k in range(self.cfg.train_samples):
            y_hat = self.mask[t] * self.model(x)
            pred_probs += y_hat.softmax(dim=-1)
            nll += cross_entropy(y_hat, y)

        pred_probs /= self.cfg.train_samples
        nll /= self.cfg.train_samples

        kl_encoder = kl_divergence(encoder)
        kl_head = 0
        if isinstance(head, VariationalLinear):
            # Only calculate KL divergence for the active weights
            kl_head = (self.mask[t] * head.weight.kl_divergences().sum(1)).sum()
            kl_head += (self.mask[t] * head.bias.kl_divergences()).sum()

        # Scale the KL divergence by the number of samples in the dataset.
        # The idea is that the cross-entropy loss is an average over a mini-batch and
        # is therefore scaled down by 1/dataset_size. To balance this, we scale the KL
        # divergence by the dataset size.
        kl = (kl_encoder + kl_head) / len(self.experience.dataset)  # type: ignore
        beta_kl = self.beta * kl
        loss = nll + beta_kl

        step = self.clock.train_iterations
        self.writer.add_scalar("ball/nll", nll.item(), step)
        self.writer.add_scalar("ball/kl", kl.item(), step)
        self.writer.add_scalar("ball/beta_kl", beta_kl.item(), step)
        return loss, pred_probs  # type: ignore

    def predict_step(
        self, batch: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        assert self.is_eval
        x, y, _ = batch

        # Output logits shape: (test_samples, ensemble_size, batch_size, num_classes)
        logits = torch.stack(
            [
                ensemble_predict(x, self.model, self.cfg.config.ensemble_size)
                for _ in range(self.cfg.test_samples)
            ]
        )

        if self.cfg.softmax_avg:
            # Softmax before averaging
            pred_probs = logits.softmax(dim=-1).mean(dim=(0, 1))
        else:
            # Average logits before softmax
            avg_logits = logits.mean(dim=(0, 1))
            pred_probs = avg_logits.softmax(dim=-1)

        loss = nll_loss(pred_probs.log(), y)
        return loss, pred_probs

    def _before_training_exp(self, **kwargs):
        # Reset the momentum optimizer to avoid forgetting previous tasks because of
        # stale momentum.
        self.optimizer = self.optimizer_fn()  # type: ignore
        return super()._before_training_exp(**kwargs)

    def _before_training_epoch(self, **kwargs):
        sigma_sum: float = 0.0
        sigma_count: int = 0
        for name, module in self.model.named_modules():
            if isinstance(module, VariationalParameter):
                sigma = module.sigma()
                sigma_sum += sigma.sum().item()
                sigma_count += sigma.numel()

        assert sigma_count > 0, "No variational parameters found in the model."
        avg_sigma = sigma_sum / sigma_count
        step = self.clock.train_iterations

        self.writer.add_scalar("ball/avg_sigma", avg_sigma, step)
        self.writer.add_scalar("ball/beta", self.beta, step)
        return super()._before_training_epoch(**kwargs)

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
            loss, mb_output = self.training_step(self.mbatch)  # type: ignore
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
            self.loss, self.mb_output = self.predict_step(self.mbatch)  # type: ignore
            self._after_eval_forward(**kwargs)

            self._after_eval_iteration(**kwargs)

    def _after_training_exp(self, **kwargs):
        logger.info("Using previous posterior as new prior.")
        posterior_to_prior(self.model.model.vit)  # type: ignore
        return super()._after_training_exp(**kwargs)
