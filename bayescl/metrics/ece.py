from avalanche.evaluation.metric_definitions import PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
import torch
from avalanche.training.templates import SupervisedTemplate
from torchmetrics.classification import (
    MulticlassCalibrationError as TorchMulticlassCalibrationError,
)

class PerExperienceBrier(PluginMetric[float]):
    def __init__(self) -> None:
        super().__init__()
        self.brier_sum = torch.tensor(0.0)
        self.sample_count = 0

    def before_eval(self, strategy) -> None:
        self.reset()

    def after_eval_iteration(self, strategy) -> None:
        self.update(strategy.mb_output, strategy.mb_y)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probabilities = logits.softmax(dim=1)
        one_hot = torch.nn.functional.one_hot(
            targets, num_classes=probabilities.shape[1]
        ).to(probabilities.dtype)
        self.brier_sum = self.brier_sum.to(probabilities.device)
        self.brier_sum += (probabilities - one_hot).square().sum()
        self.sample_count += targets.numel()

    def after_eval_exp(self, strategy) -> MetricResult:
        result = [MetricValue(self, "brier", self.compute(), 0)]
        self.reset()
        return result

    def compute(self) -> float:
        return (self.brier_sum / self.sample_count).item()

    def result(self) -> float | None:
        return None

    def reset(self) -> None:
        self.brier_sum = torch.tensor(0.0)
        self.sample_count = 0

    def __str__(self) -> str:
        return "Brier"


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
        prefix = f"ECE/{task:02d}/"
        result.append(
            MetricValue(self, prefix + "all", self.ece_all.compute().item(), i)
        )

        # Empty at the end of the final task
        if len(self.ece_present.confidences) != 0:
            result.append(
                MetricValue(
                    self, prefix + "present", self.ece_present.compute().item(), i
                )
            )

        # Empty if no past tasks
        if len(self.ece_past.confidences) != 0:
            result.append(
                MetricValue(self, prefix + "past", self.ece_past.compute().item(), i)
            )

        # Empty if no future tasks
        if len(self.ece_future.confidences) != 0:
            result.append(
                MetricValue(
                    self, prefix + "future", self.ece_future.compute().item(), i
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
