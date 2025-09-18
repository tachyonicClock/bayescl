import torch
from avalanche.evaluation.metric_definitions import PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.training.templates import SupervisedTemplate
from torchmetrics.classification import (
    MulticlassCalibrationError as TorchMulticlassCalibrationError,
)


class ExpectedCalibrationError(PluginMetric[float]):
    def __init__(self, num_classes: int, **kwargs) -> None:
        super().__init__()
        self.metric = TorchMulticlassCalibrationError(num_classes, **kwargs)
        self.y_pred = []
        self.y_true = []
        self.y_conf = []
        self.eval_exp_counter = 0

    def before_eval(self, strategy) -> None:
        self.eval_exp_counter = 0

    def before_eval_exp(self, strategy: SupervisedTemplate) -> None:
        self.eval_exp_counter += 1

    def after_eval_iteration(self, strategy) -> None:
        # Only calculate ECE on tasks seen so far
        if self.eval_exp_counter <= strategy.clock.train_exp_counter:
            self.metric.update(strategy.mb_output, strategy.mb_y)  # type: ignore

            y_proba = torch.nn.functional.softmax(strategy.mb_output, dim=1)
            y_conf, y_pred = torch.max(y_proba, dim=1)
            self.y_true.append(strategy.mb_y.cpu())
            self.y_pred.append(y_pred.cpu())
            self.y_conf.append(y_conf.cpu())

    def after_eval(self, strategy) -> MetricResult:
        i = strategy.clock.train_iterations
        metric_value = MetricValue(self, str(self), self.result(), i)
        self.reset()
        return [metric_value]

    def result(self) -> float:
        return self.metric.compute().item()

    def reset(self) -> None:
        self.metric.reset()
        self.y_pred = []
        self.y_true = []
        self.y_conf = []

    def __str__(self) -> str:
        return "ECE"
