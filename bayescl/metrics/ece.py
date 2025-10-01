from avalanche.evaluation.metric_definitions import PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.training.templates import SupervisedTemplate
from torchmetrics.classification import (
    MulticlassCalibrationError as TorchMulticlassCalibrationError,
)


class ExpectedCalibrationError(PluginMetric[float]):
    def __init__(self, num_classes: int, **kwargs) -> None:
        super().__init__()
        self.ece_seen = TorchMulticlassCalibrationError(num_classes, **kwargs)
        self.ece_all = TorchMulticlassCalibrationError(num_classes, **kwargs)
        self.ece_unseen = TorchMulticlassCalibrationError(num_classes, **kwargs)
        self.eval_exp_counter = 0

    def before_eval(self, strategy) -> None:
        self.eval_exp_counter = 0

    def before_eval_exp(self, strategy: SupervisedTemplate) -> None:
        self.eval_exp_counter += 1

    def after_eval_iteration(self, strategy) -> None:
        self.ece_all.update(strategy.mb_output, strategy.mb_y)  # type: ignore

        # Only calculate ECE on tasks seen so far
        if self.eval_exp_counter <= strategy.clock.train_exp_counter:
            self.ece_seen.update(strategy.mb_output, strategy.mb_y)  # type: ignore
        else:
            self.ece_unseen.update(strategy.mb_output, strategy.mb_y)  # type: ignore

    def after_eval(self, strategy) -> MetricResult:
        i = strategy.clock.train_iterations
        ece_seen = MetricValue(self, "ECE/seen", self.ece_seen.compute().item(), i)
        ece_all = MetricValue(self, "ECE/all", self.ece_all.compute().item(), i)
        ece_unseen = MetricValue(
            self, "ECE/unseen", self.ece_unseen.compute().item(), i
        )
        return [ece_seen, ece_all, ece_unseen]

    def result(self) -> float | None:
        return None

    def reset(self) -> None:
        self.ece_seen.reset()
        self.ece_all.reset()
        self.ece_unseen.reset()

    def __str__(self) -> str:
        return "ECE"
