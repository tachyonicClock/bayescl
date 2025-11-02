from typing import Any

from avalanche.training.plugins import SupervisedPlugin

from bayescl.metrics.results import ContinualLearningEvaluator


class MetricsPlugin(SupervisedPlugin):
    def __init__(self, num_tasks: int, num_classes: int) -> None:
        self.evaluator = ContinualLearningEvaluator(num_tasks, num_classes)
        self._eval_task = 0
        self._train_task = -1
        self._is_final = False

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
        if self._is_final:
            _, y_true, _ = strategy.mbatch
            self.evaluator.update(
                train_task_idx=self._train_task,
                test_task_idx=self._eval_task,
                y_logit=strategy.mb_output,
                y=y_true,
            )
