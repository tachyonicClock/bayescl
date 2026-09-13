import copy
from typing import Any, Sequence

from avalanche.training.plugins import SupervisedPlugin

from bayescl.metrics.results import ContinualLearningEvaluator


class BrierEarlyStopping(SupervisedPlugin):
    """Stops training the current task once its held-out validation Brier
    score stops improving, restoring the best-checkpoint weights either way.

    Runs its own periodic validation pass every ``eval_every`` epochs through
    ``eval_and_capture`` (``Experiment._eval_and_capture``) so these extra
    eval passes never reach the main evaluator's seen/future-task
    bookkeeping -- the same isolation mechanism used for OOD/shift eval.
    """

    def __init__(
        self,
        eval_and_capture,
        val_stream: Sequence[Any],
        loader_kwargs: dict,
        eval_every: int,
        patience: int,
    ) -> None:
        self._eval_and_capture = eval_and_capture
        self.val_stream = val_stream
        self.loader_kwargs = loader_kwargs
        self.eval_every = eval_every
        self.patience = patience
        self._task_idx = -1
        # Tracked locally rather than read off ``strategy.clock.train_exp_epochs``:
        # avalanche's ``Clock`` plugin must run last to keep its counters correct
        # for plugins after it, and this plugin isn't guaranteed that position.
        self._epoch_in_task = 0
        self._best_brier: float | None = None
        self._epochs_without_improvement = 0
        self._best_state: dict | None = None

    def before_training_exp(self, strategy: Any, *args, **kwargs) -> None:
        self._task_idx += 1
        self._epoch_in_task = 0
        self._best_brier = None
        self._epochs_without_improvement = 0
        self._best_state = None
        self._check(strategy)

    def before_training_epoch(self, strategy: Any, *args, **kwargs) -> None:
        if self._epochs_without_improvement >= self.patience:
            if self._best_state is not None:
                strategy.model.load_state_dict(self._best_state)
            strategy.stop_training()

    def after_training_epoch(self, strategy: Any, *args, **kwargs) -> None:
        self._epoch_in_task += 1
        if self._epoch_in_task % self.eval_every == 0:
            self._check(strategy)

    def after_training_exp(self, strategy: Any, *args, **kwargs) -> None:
        if self._best_state is not None:
            strategy.model.load_state_dict(self._best_state)

    def _check(self, strategy: Any) -> None:
        logits, y = self._eval_and_capture(
            strategy, self.val_stream[self._task_idx], self.loader_kwargs
        )
        brier = ContinualLearningEvaluator.brier(logits, y)
        if self._best_brier is None or brier < self._best_brier:
            self._best_brier = brier
            self._epochs_without_improvement = 0
            self._best_state = copy.deepcopy(strategy.model.state_dict())
        else:
            self._epochs_without_improvement += 1
