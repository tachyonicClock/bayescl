from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

from avalanche.training.plugins import SupervisedPlugin
from torch import Tensor

from bayescl.metrics.results import ContinualLearningEvaluator


class MetricsPlugin(SupervisedPlugin):
    def __init__(self, num_tasks: int, num_classes: int) -> None:
        self.evaluator = ContinualLearningEvaluator(num_tasks, num_classes)
        self._eval_task = 0
        self._train_task = -1
        self._is_final = False
        self._capture: Optional[Callable[[Tensor, Tensor], None]] = None

    def before_training_exp(self, strategy: Any, *args, **kwargs) -> Any:
        self._train_task += 1
        self._is_final = False

    def after_training_exp(self, strategy: Any, *args, **kwargs) -> Any:
        self._is_final = True

    def before_eval(self, strategy: Any, *args, **kwargs) -> Any:
        self._eval_task = 0

    def after_eval_exp(self, strategy: Any, *args, **kwargs) -> Any:
        self._eval_task += 1

    def after_eval_iteration(self, strategy: Any, *args, **kwargs) -> Any:
        _, y_true, _ = strategy.mbatch
        if self._capture is not None:
            self._capture(strategy.mb_output, y_true)
            return
        if self._is_final:
            self.evaluator.update(
                train_task_idx=self._train_task,
                test_task_idx=self._eval_task,
                y_logit=strategy.mb_output,
                y=y_true,
            )

    @contextmanager
    def capture(self, sink: Callable[[Tensor, Tensor], None]) -> Iterator[None]:
        """Divert ``(logits, y)`` from every eval iteration to ``sink`` instead
        of the main evaluator, for the duration of the ``with`` block.

        Used to run ad-hoc ``strategy.eval(...)`` passes (OOD/shift datasets)
        through the model without polluting the clean-data bookkeeping used
        for ``acc``/``ece``/``ace``/``brier``/``nll``/``auroc_future``.
        """
        previous = self._capture
        self._capture = sink
        try:
            yield
        finally:
            self._capture = previous
