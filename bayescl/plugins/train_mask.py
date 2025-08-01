from typing import Any

from avalanche.training.plugins import SupervisedPlugin
from claiutil.datasets import avalanche_class_schedule, class_schedule_to_task_mask


class TrainTaskMask(SupervisedPlugin):
    def __init__(self, benchmark: Any):
        super().__init__()
        self.mask = class_schedule_to_task_mask(
            avalanche_class_schedule(benchmark), benchmark.n_classes
        )
        print(self.mask)

    def before_training(self, strategy: Any, *args, **kwargs) -> Any:
        self.mask = self.mask.to(strategy.device)

    def after_forward(self, strategy: Any, *args, **kwargs) -> Any:
        t = strategy.clock.train_exp_counter
        strategy.mb_output = self.mask[t] * strategy.mb_output
