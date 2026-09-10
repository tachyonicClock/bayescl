import matplotlib

matplotlib.use("Agg")

import json
import pickle
import random
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Sequence

import numpy as np
import optuna
import torch
from avalanche.evaluation.metrics import (
    StreamConfusionMatrix,
    accuracy_metrics,
    forgetting_metrics,
    loss_metrics,
    timing_metrics,
)
from avalanche.logging import BaseLogger, InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from loguru import logger
from optuna import Trial
from setproctitle import setproctitle
from torch import BoolTensor

from bayescl.benchmark import get_benchmark
from bayescl.config import ExperimentConfig
from bayescl.metrics.ece import (
    ExpectedCalibrationError,
)
from bayescl.metrics.plugin import MetricsPlugin
from bayescl.model import get_model
from bayescl.peft import parameter_summary_str

__all__ = ["Experiment", "ExperimentConfig"]


def avalanche_class_schedule(
    benchmark: Any,
) -> Sequence[set[int]]:
    schedule = []
    for task in benchmark.train_stream:
        schedule.append(set(task.classes_in_this_experience))
    return schedule


def class_schedule_to_task_mask(
    class_schedule: Sequence[set[int]], num_classes: int
) -> BoolTensor:
    """Convert a class schedule to a list of boolean masks.

    This is useful when implementing multi-headed neural networks for task
    incremental learning.

    >>> class_schedule_to_task_mask([{0, 1}, {2, 3}], 4)
    tensor([[ True,  True, False, False],
            [False, False,  True,  True]])

    :param num_classes: The total number of classes.
    :param class_schedule: A sequence of sets containing class indices defining
        task order and composition.
    :return: A boolean mask of shape (num_tasks, num_classes)
    """
    min_class = min(map(min, class_schedule), default=-1)
    max_class = max(map(max, class_schedule), default=-1)
    if not 0 <= min_class < num_classes or not 0 <= max_class < num_classes:
        raise ValueError(
            "Classes in the schedule should be within the range of num_classes"
        )

    task_mask = torch.zeros(len(class_schedule), num_classes, dtype=torch.bool)
    for i, classes in enumerate(class_schedule):
        task_mask[i, list(classes)] = True
    return BoolTensor(task_mask)


class Experiment:
    def _new_eval_plugin(self) -> EvaluationPlugin:
        return EvaluationPlugin(
            accuracy_metrics(
                minibatch=False,
                epoch=True,
                experience=True,
                stream=True,
                trained_experience=True,
            ),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            timing_metrics(epoch=True),
            forgetting_metrics(experience=True, stream=True),
            StreamConfusionMatrix(
                num_classes=self.benchmark.n_classes, save_image=True
            ),
            ExpectedCalibrationError(self.num_classes),
            loggers=self.loggers,
        )

    def _prepare_run_dir(self) -> Path:
        run_dir = self.config.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        setproctitle(f"bayescl.{self.config.dataset}")
        with open(run_dir / "spec.json", "w") as f:
            json.dump(self._config(), f, indent=2, default=str)
        logger.info(f"Logging to '{run_dir}'")
        return run_dir

    def _new_logger(self) -> TensorboardLogger:
        tb_logger = TensorboardLogger(self.config.run_dir)
        self.loggers.append(InteractiveLogger())
        self.loggers.append(tb_logger)
        return tb_logger

    def _preflight(self):
        logger.info("Resolved Spec:\n{}", pformat(self._config()))
        logger.info("Parameter Counts:\n{}", parameter_summary_str(self.model))
        logger.info("Plugins:\n{}", [type(p).__name__ for p in self.plugins])

    def _seed_everything(self):
        if self.config.seed is not None:
            logger.info(f"Set random seed to {self.config.seed}")
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

    def __init__(
        self,
        config: ExperimentConfig,
        arm,
    ) -> None:
        self.config = config
        self.arm = arm
        self._seed_everything()
        self.plugins: List[SupervisedPlugin] = []
        self.loggers: List[BaseLogger] = []

        self.benchmark = get_benchmark(self.config)
        self.mask = class_schedule_to_task_mask(
            avalanche_class_schedule(self.benchmark), self.benchmark.n_classes
        )
        self.num_tasks: int = len(self.benchmark.train_stream)
        self.num_classes: int = self.benchmark.n_classes
        self.run_dir: Path = self._prepare_run_dir()
        self.tb_log = self._new_logger()
        self.eval_plugin: EvaluationPlugin = self._new_eval_plugin()
        self.metrics_plugin = MetricsPlugin(self.num_tasks, self.num_classes)
        self.model = get_model(self.config, self.benchmark.n_classes)
        self.arm._build_peft(self)
        self.arm._build_plugins(self)
        self.arm.build(self)

    def _config(self) -> dict:
        return asdict(self.config)

    def run(
        self, trial: Trial | None = None, *, report_intermediate: bool = True
    ) -> tuple[float, float]:
        self._preflight()
        strategy = self.arm._build_strategy(self)
        strategy.mask = self.mask.to(self.config.device)  # type: ignore

        # TRAINING LOOP
        logger.info("Starting experiment...")
        results: Sequence[Dict[str, float]] = []
        for t, experience in enumerate(self.benchmark.train_stream):
            logger.info(f"Start of experience: {experience.current_experience}")
            logger.info(f"Experience Size: {len(experience.dataset)}")
            logger.info(f"Current Classes: {experience.classes_in_this_experience}")

            strategy.train_epochs = self.config.epochs

            # ``persistent_workers`` avoids respawning the worker pool every epoch;
            # ``pin_memory`` speeds up the host->GPU copy. Both are forwarded by
            # Avalanche to the underlying ``DataLoader``.
            loader_kwargs = dict(
                num_workers=self.config.num_workers,
                pin_memory=True,
                persistent_workers=self.config.num_workers > 0,
            )

            # train returns a dictionary which contains all the metric values
            strategy.train(
                experience,
                self.benchmark.test_stream[: t + 1],
                **loader_kwargs,
            )

            results.append(
                strategy.eval(self.benchmark.test_stream, **loader_kwargs)
            )
            if trial is not None and report_intermediate:
                intermediate_acc, intermediate_ece = (
                    self.metrics_plugin.evaluator.intermediate_result(t)
                )
                intermediate_score = 0.5 * (intermediate_acc + (1 - intermediate_ece))
                trial.report(intermediate_score, step=t)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        # Save results to run directory
        with open(self.config.run_dir / "avalanche_results.pkl", "wb") as f:
            pickle.dump(results, f)

        metrics, raw_data = self.metrics_plugin.evaluator.result()
        pickle.dump(metrics, open(self.config.run_dir / "metrics.pkl", "wb"))
        pickle.dump(raw_data, open(self.config.run_dir / "raw_data.pkl", "wb"))

        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                logger.info(f"{key}: {value:.4f}")
        return metrics["accuracy_seen_avg"], metrics["ece_seen_avg"]

    def count_parameters(self):
        print(parameter_summary_str(self.model))
