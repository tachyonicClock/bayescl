import copy
from typing import Any, Sequence

from avalanche.training.plugins import SupervisedPlugin
from loguru import logger

from bayescl.base import ConvergenceError
from bayescl.metrics.results import ContinualLearningEvaluator, normalize_predictive


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
        strict: bool = False,
        writer: Any = None,
    ) -> None:
        self._eval_and_capture = eval_and_capture
        self.val_stream = val_stream
        self.loader_kwargs = loader_kwargs
        self.eval_every = eval_every
        self.patience = patience
        #: Optional ``SummaryWriter``. The probe's scores are the only
        #: per-epoch view of held-out performance, so they're worth a curve;
        #: left optional so the plugin stays constructible in tests.
        self._writer = writer
        #: If True, a task that exhausts its epoch budget without early
        #: stopping ever triggering raises ``ConvergenceError`` instead of
        #: silently completing -- used during hyperparameter tuning so
        #: Optuna can't select a "best" trial that was actually just cut off
        #: mid-improvement rather than genuinely converged (see
        #: ``Experiment.__init__``, which only sets this during tuning).
        self.strict = strict
        self._task_idx = -1
        # Tracked locally rather than read off ``strategy.clock.train_exp_epochs``:
        # avalanche's ``Clock`` plugin must run last to keep its counters correct
        # for plugins after it, and this plugin isn't guaranteed that position.
        self._epoch_in_task = 0
        self._best_brier: float | None = None
        self._epochs_without_improvement = 0
        self._best_state: dict | None = None
        #: The epoch each finished task's training exited at (whether by
        #: early stopping or exhausting the epoch budget), one entry per task.
        self.exit_epochs: list[int] = []

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
        self.exit_epochs.append(self._epoch_in_task)
        if self.strict and self._epoch_in_task >= strategy.train_epochs:
            raise ConvergenceError(
                f"task {self._task_idx} did not converge within the "
                f"{strategy.train_epochs}-epoch budget (early stopping never "
                "triggered)"
            )

    def _check(self, strategy: Any) -> None:
        logits, y = self._eval_and_capture(
            strategy, self.val_stream[self._task_idx], self.loader_kwargs
        )
        probs = normalize_predictive(logits)
        brier = ContinualLearningEvaluator.brier(probs, y)
        ece = ContinualLearningEvaluator.ece(probs, y)
        # Accuracy and NLL are computed here rather than read off the metric
        # stream: this pass evaluates a single task's validation split, so the
        # stream-level metrics describe something other than what their names
        # say and are muted for its duration.
        accuracy = float((probs.argmax(dim=1) == y).double().mean())
        nll = ContinualLearningEvaluator.nll(probs, y)
        epoch = max(self._epoch_in_task - 1, 0)
        with logger.contextualize(eval_tag="early_stop"):
            logger.info(
                f"task={self._task_idx} epoch={epoch} | "
                f"val_accuracy={accuracy:.4f} val_nll={nll:.4f} "
                f"brier={brier:.4f} ece={ece:.4f}"
            )
        if self._writer is not None:
            for name, value in (
                ("accuracy", accuracy),
                ("nll", nll),
                ("brier", brier),
                ("ece", ece),
            ):
                self._writer.add_scalar(
                    f"val/task_{self._task_idx:02d}/{name}", value, epoch
                )
        if self._best_brier is None or brier < self._best_brier:
            self._best_brier = brier
            self._epochs_without_improvement = 0
            self._best_state = copy.deepcopy(strategy.model.state_dict())
        else:
            self._epochs_without_improvement += 1
