from avalanche.evaluation.metric_definitions import PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.training.templates import SupervisedTemplate
from torchmetrics.classification import (
    MulticlassCalibrationError as TorchMulticlassCalibrationError,
)



class ExpectedCalibrationError(PluginMetric[float]):
    def __init__(self, num_classes: int, **kwargs) -> None:
        super().__init__()
        self.ece_trained = TorchMulticlassCalibrationError(num_classes, **kwargs)
        self.ece_unseen = TorchMulticlassCalibrationError(num_classes, **kwargs)
        #: Calibration on just the task that finished training. Unlike the
        #: other splits this one overlaps: it is a subset of ``trained``, so
        #: the gap between the two is how much calibration on older tasks has
        #: decayed relative to the freshest one.
        self.ece_last = TorchMulticlassCalibrationError(num_classes, **kwargs)
        self.ece_all = TorchMulticlassCalibrationError(num_classes, **kwargs)
        self.eval_exp_counter = 0

    def before_eval(self, strategy) -> None:
        self.eval_exp_counter = 0

    def after_eval_exp(self, strategy: SupervisedTemplate) -> None:
        self.eval_exp_counter += 1

    def after_eval_iteration(self, strategy) -> None:
        # ``train_exp_counter`` has already advanced past the task that just
        # finished training, so it is the index of the first *untrained* task:
        # everything below it has been trained on, everything from it upwards
        # has not. The splits were once past/present/future, which read as
        # though the task just finished were the "present" one -- it isn't, it
        # lands in ``trained``.
        first_untrained = strategy.clock.train_exp_counter
        self.ece_all.update(strategy.mb_output, strategy.mb_y)  # type: ignore

        if self.eval_exp_counter < first_untrained:
            self.ece_trained.update(strategy.mb_output, strategy.mb_y)  # type: ignore
        else:
            self.ece_unseen.update(strategy.mb_output, strategy.mb_y)  # type: ignore

        # Negative before any training, so this stays empty on the first probe.
        if self.eval_exp_counter == first_untrained - 1:
            self.ece_last.update(strategy.mb_output, strategy.mb_y)  # type: ignore

    def after_eval(self, strategy) -> MetricResult:
        # Stepped by training iteration and named without the task index.
        # Folding the index into the tag instead gave one single-point series
        # per task rather than one curve per split, the same way a per-task
        # Brier tag once did.
        i = strategy.clock.train_iterations
        result = []
        prefix = "ECE/"
        result.append(
            MetricValue(self, prefix + "all", self.ece_all.compute().item(), i)
        )

        # Empty before any task has been trained
        if len(self.ece_trained.confidences) != 0:
            result.append(
                MetricValue(
                    self, prefix + "trained", self.ece_trained.compute().item(), i
                )
            )

        # Empty before any task has been trained
        if len(self.ece_last.confidences) != 0:
            result.append(
                MetricValue(self, prefix + "last", self.ece_last.compute().item(), i)
            )

        # Empty at the end of the final task -- nothing is left untrained
        if len(self.ece_unseen.confidences) != 0:
            result.append(
                MetricValue(
                    self, prefix + "unseen", self.ece_unseen.compute().item(), i
                )
            )

        self.reset()
        return result

    def result(self) -> float | None:
        return None

    def reset(self) -> None:
        self.ece_trained.reset()
        self.ece_unseen.reset()
        self.ece_last.reset()
        self.ece_all.reset()
        self.eval_exp_counter = 0

    def __str__(self) -> str:
        return "ECE"
