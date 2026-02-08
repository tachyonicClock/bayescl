from avalanche.evaluation.metric_definitions import PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.training.templates import SupervisedTemplate
from loguru import logger
from torchmetrics.classification import (
    MulticlassCalibrationError as TorchMulticlassCalibrationError,
)


class ExpectedCalibrationError(PluginMetric[float]):
    def __init__(self, num_classes: int, **kwargs) -> None:
        super().__init__()
        self.ece_past = TorchMulticlassCalibrationError(num_classes, **kwargs)
        self.ece_present = TorchMulticlassCalibrationError(num_classes, **kwargs)
        self.ece_future = TorchMulticlassCalibrationError(num_classes, **kwargs)
        self.ece_all = TorchMulticlassCalibrationError(num_classes, **kwargs)
        self.eval_exp_counter = 0

    def before_eval(self, strategy) -> None:
        self.eval_exp_counter = 0

    def after_eval_exp(self, strategy: SupervisedTemplate) -> None:
        self.eval_exp_counter += 1
        logger.info(f"{self.eval_exp_counter} {strategy.clock.train_exp_counter}")

    def after_eval_iteration(self, strategy) -> None:
        # ECE all
        task = strategy.clock.train_exp_counter
        self.ece_all.update(strategy.mb_output, strategy.mb_y)  # type: ignore

        # ECE past
        if self.eval_exp_counter < task:
            self.ece_past.update(strategy.mb_output, strategy.mb_y)  # type: ignore

        # ECE present
        if self.eval_exp_counter == task:
            self.ece_present.update(strategy.mb_output, strategy.mb_y)  # type: ignore

        # ECE future
        if self.eval_exp_counter > task:
            self.ece_future.update(strategy.mb_output, strategy.mb_y)  # type: ignore

    def after_eval(self, strategy) -> MetricResult:
        task = strategy.clock.train_exp_counter
        i = strategy.clock.train_iterations
        result = []
        result.append(
            MetricValue(self, f"ECE/{task}/all", self.ece_all.compute().item(), i)
        )

        # Empty at the end of the final task
        if len(self.ece_present.confidences) != 0:
            result.append(
                MetricValue(
                    self,
                    f"ECE/{task:02d}/present",
                    self.ece_present.compute().item(),
                    i,
                )
            )

        # Empty if no past tasks
        if len(self.ece_past.confidences) != 0:
            result.append(
                MetricValue(
                    self, f"ECE/{task:02d}/past", self.ece_past.compute().item(), i
                )
            )

        # Empty if no future tasks
        if len(self.ece_future.confidences) != 0:
            result.append(
                MetricValue(
                    self, f"ECE/{task:02d}/future", self.ece_future.compute().item(), i
                )
            )

        self.reset()
        return result

    def result(self) -> float | None:
        return None

    def reset(self) -> None:
        self.ece_past.reset()
        self.ece_present.reset()
        self.ece_future.reset()
        self.ece_all.reset()
        self.eval_exp_counter = 0

    def __str__(self) -> str:
        return "ECE"
