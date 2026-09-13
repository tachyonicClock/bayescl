from typing import Any, Callable, Tuple

import torch
from avalanche.training.supervised import Naive
from torch import BoolTensor, Tensor
from torch.nn.functional import nll_loss


class EnsembleStrategy(Naive):
    """Trains ``num_members`` independent models on the same task stream.

    Each member has its own optimizer and is trained with its own
    per-task-masked cross-entropy loss -- there is no shared objective, so
    this is a standard deep ensemble rather than a single jointly-optimised
    model. Predictions are combined only at evaluation time, by averaging
    per-member softmax probabilities (log-mean-exp of log-softmax), which is
    the classic deep-ensemble predictive distribution.
    """

    def __init__(
        self,
        *,
        mask: BoolTensor,
        optimizer_fn: Callable[[Any], torch.optim.Optimizer],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._optimizer_fn = optimizer_fn
        self._mask = mask.to(self.device)
        self._member_optimizers: list[torch.optim.Optimizer] = []

    def _before_training_exp(self, **kwargs):
        # Fresh optimizers per task avoid carrying stale momentum forward,
        # matching how the other LoRA-based arms reset ``strategy.optimizer``
        # in ``TrainTaskMask.before_training_exp``.
        self._member_optimizers = [
            self._optimizer_fn(member.parameters()) for member in self.model.members
        ]
        return super()._before_training_exp(**kwargs)

    def training_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self._before_forward(**kwargs)
            self.mb_output, self.loss = self._training_step(self.mbatch)
            self._after_forward(**kwargs)

            # Backward/optimizer steps already happened per-member inside
            # `_training_step`, since each member needs its own gradients
            # and optimizer rather than a single shared update.
            self._before_backward(**kwargs)
            self._after_backward(**kwargs)
            self._before_update(**kwargs)
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def _training_step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        mask = self._mask[self.clock.train_exp_counter]
        x, y, _ = batch

        outputs = []
        losses = []
        for member, optimizer in zip(self.model.members, self._member_optimizers):
            optimizer.zero_grad()
            out = mask * member(x)
            loss = self._criterion(out, y)
            loss.backward()
            optimizer.step()
            outputs.append(out.detach())
            losses.append(loss.detach())

        mean_probs = torch.stack(outputs).softmax(dim=-1).mean(dim=0)
        log_mean_probs = mean_probs.clamp_min(1e-12).log()
        return log_mean_probs, torch.stack(losses).mean()

    def eval_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)
            self.mb_output, self.loss = self._eval_step(self.mbatch)
            self._after_eval_forward(**kwargs)

            self._after_eval_iteration(**kwargs)

    def _eval_step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x, y, _ = batch
        outputs = torch.stack([member(x) for member in self.model.members])
        mean_probs = outputs.softmax(dim=-1).mean(dim=0)
        log_mean_probs = mean_probs.clamp_min(1e-12).log()
        return log_mean_probs, nll_loss(log_mean_probs, y)
