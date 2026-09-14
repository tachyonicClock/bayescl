import matplotlib

matplotlib.use("Agg")

import json
import pickle
import random
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Sequence

import numpy as np
import optuna
import torch
from avalanche.benchmarks.scenarios.deprecated.generic_benchmark_creation import (
    create_multi_dataset_generic_benchmark,
)
from avalanche.evaluation.metrics import (
    StreamConfusionMatrix,
    accuracy_metrics,
    forgetting_metrics,
    loss_metrics,
    timing_metrics,
)
from avalanche.logging import BaseLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from loguru import logger
from optuna import Trial
from setproctitle import setproctitle
from torch import BoolTensor
from torch.utils.data import ConcatDataset

from bayescl.config import ExperimentConfig
from bayescl.data.benchmark import (
    ShiftedTensorDataset,
    eval_transform_for,
    get_benchmark,
)
from bayescl.data.datasets import SHIFT_SEVERITIES, get_ood_dataset, ood_dataset_names
from bayescl.metrics.agent_logger import AgentLogger, set_log_file
from bayescl.metrics.ece import ExpectedCalibrationError, PerExperienceBrier
from bayescl.metrics.plugin import MetricsPlugin
from bayescl.metrics.results import Result
from bayescl.model import get_model
from bayescl.peft import parameter_summary_str
from bayescl.plugin.brier_early_stopping import BrierEarlyStopping

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
            PerExperienceBrier(),
            loggers=self.loggers,
        )

    def _prepare_run_dir(self) -> Path:
        run_dir = self.config.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        setproctitle(f"bayescl.{self.config.dataset}")
        with open(run_dir / "spec.json", "w") as f:
            json.dump(self._config(), f, indent=2, default=str)
        set_log_file(run_dir / "run.log")
        logger.info(f"Logging to '{run_dir}'")
        return run_dir

    def _new_logger(self) -> TensorboardLogger:
        tb_logger = TensorboardLogger(self.config.run_dir)
        self.loggers.append(AgentLogger())
        self.loggers.append(tb_logger)
        return tb_logger

    def _preflight(self):
        logger.info("Parameter Counts:\n{}", parameter_summary_str(self.model))
        logger.info("Resolved Spec: {}", self._config())
        logger.info("Plugins:{}", [type(p).__name__ for p in self.plugins])

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
        self.plugins: list[SupervisedPlugin] = []
        self.loggers: list[BaseLogger] = []

        self.benchmark = get_benchmark(self.config)
        self.mask = class_schedule_to_task_mask(
            avalanche_class_schedule(self.benchmark), self.benchmark.n_classes
        )
        self.num_tasks: int = len(self.benchmark.train_stream)
        self.num_classes: int = self.benchmark.n_classes
        self.run_dir: Path = self._prepare_run_dir()
        self.tb_log = self._new_logger()
        self.eval_plugin: EvaluationPlugin = self._new_eval_plugin()
        self.metrics_plugin: MetricsPlugin = MetricsPlugin(
            self.num_tasks, self.num_classes
        )
        # ``persistent_workers`` avoids respawning the worker pool every epoch;
        # ``pin_memory`` speeds up the host->GPU copy. Both are forwarded by
        # Avalanche to the underlying ``DataLoader``.
        self.loader_kwargs = dict(
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )
        self.model = get_model(self.config, self.benchmark.n_classes)
        self.arm._build_peft(self)
        self.arm._build_plugins(self)
        self.arm.build(self)
        self.plugins.append(
            BrierEarlyStopping(
                partial(self._eval_and_capture, tag="early_stop"),
                self._val_stream(),
                self.loader_kwargs,
                eval_every=self.config.early_stop_eval_every,
                patience=self.config.early_stop_patience,
            )
        )

    def _config(self) -> dict:
        return asdict(self.config)

    def _val_stream(self) -> Sequence[Any]:
        """Per-task held-out validation stream used for early stopping.

        Kept independent of ``self.benchmark`` so early stopping is always
        driven by validation data, never by the scale's test split, even
        when this run's own eval stream (``self.benchmark.test_stream``) is
        that test split.
        """
        if self.config.validation:
            return self.benchmark.test_stream
        return get_benchmark(replace(self.config, validation=True)).test_stream

    def _ood_test_streams(self) -> Dict[str, Any]:
        """One-experience test stream per auxiliary OOD dataset."""
        eval_transform = eval_transform_for(self.config)
        streams = {}
        for name in ood_dataset_names():
            ood_dataset = get_ood_dataset(
                name, self.config.scale, self.config.dataset_root, eval_transform
            )
            benchmark = create_multi_dataset_generic_benchmark(
                train_datasets=[ood_dataset],
                test_datasets=[ood_dataset],
                complete_test_set_only=True,
            )
            streams[name] = benchmark.test_stream
        return streams

    def _shift_test_stream(self, severity: int, seen_task_count: int) -> Any:
        """One-experience test stream of seen-task samples corrupted at
        ``severity``, the ``ece@$shift``/``ace@$shift`` eval data."""
        shifted = [
            ShiftedTensorDataset(
                self.benchmark.test_stream[i].dataset, severity, self.config.standardize
            )
            for i in range(seen_task_count)
        ]
        combined = shifted[0] if len(shifted) == 1 else ConcatDataset(shifted)
        benchmark = create_multi_dataset_generic_benchmark(
            train_datasets=[combined],
            test_datasets=[combined],
            complete_test_set_only=True,
        )
        return benchmark.test_stream

    def _eval_and_capture(
        self, strategy, stream, loader_kwargs: dict, *, tag: str
    ) -> tuple:
        """Run ``strategy.eval(stream)`` without touching the main evaluator's
        clean-data bookkeeping; returns the pooled ``(logits, y)``.

        ``tag`` is attached to the eval's log lines (via ``logger.contextualize``)
        so these auxiliary passes -- early-stopping probes, OOD/shift eval --
        are distinguishable from the real per-checkpoint eval on
        ``self.benchmark.test_stream``, which doesn't go through this method.
        """
        buffer: list[tuple] = []
        with (
            self.metrics_plugin.capture(
                lambda logits, y, buf=buffer: buf.append(
                    (logits.detach().cpu(), y.detach().cpu())
                )
            ),
            logger.contextualize(eval_tag=tag),
        ):
            strategy.eval(stream, **loader_kwargs)
        logits = torch.cat([logit for logit, _ in buffer], dim=0)
        y = torch.cat([label for _, label in buffer], dim=0)
        return logits, y

    def run(
        self, trial: Trial | None = None, *, report_intermediate: bool = True
    ) -> Result:
        self._preflight()
        strategy = self.arm._build_strategy(self)
        strategy.mask = self.mask.to(self.config.device)  # type: ignore

        # OOD/shift metrics (auroc_$ood_dataset, ece@$shift, ace@$shift) require
        # extra eval passes per checkpoint (1 per OOD dataset + 1 per shift
        # severity) on top of the normal seen/future eval -- only worth paying
        # for on the final ``test`` run, not on every ``tune`` HPO trial.
        compute_shift_ood_metrics = trial is None
        if compute_shift_ood_metrics:
            logger.info(
                "Computing OOD ({}) and shift (severities {}) metrics every "
                "checkpoint -- multiplies per-checkpoint eval cost.",
                ood_dataset_names(),
                SHIFT_SEVERITIES,
            )
        ood_streams = self._ood_test_streams() if compute_shift_ood_metrics else {}

        # TRAINING LOOP
        results: Sequence[Dict[str, float]] = []
        for t, experience in enumerate(self.benchmark.train_stream):
            logger.info(f"Start of experience: {experience.current_experience}")
            logger.info(f"Experience Size: {len(experience.dataset)}")
            logger.info(f"Current Classes: {experience.classes_in_this_experience}")

            strategy.train_epochs = self.config.epochs

            # train returns a dictionary which contains all the metric values
            strategy.train(
                experience,
                self.benchmark.test_stream[: t + 1],
                **self.loader_kwargs,
            )

            results.append(
                strategy.eval(self.benchmark.test_stream, **self.loader_kwargs)
            )
            brier_all, brier_seen = (
                self.metrics_plugin.evaluator.checkpoint_brier_scores(t)
            )
            self.tb_log.writer.add_scalar(f"Brier/{t:02d}/all", brier_all, t)
            self.tb_log.writer.add_scalar(f"Brier/{t:02d}/seen", brier_seen, t)
            if compute_shift_ood_metrics:
                for name, stream in ood_streams.items():
                    logits, _ = self._eval_and_capture(
                        strategy, stream, self.loader_kwargs, tag=f"ood:{name}"
                    )
                    self.metrics_plugin.evaluator.record_ood(name, t, logits)

                for severity in SHIFT_SEVERITIES:
                    stream = self._shift_test_stream(severity, t + 1)
                    logits, y = self._eval_and_capture(
                        strategy, stream, self.loader_kwargs, tag=f"shift:{severity}"
                    )
                    self.metrics_plugin.evaluator.record_shift(severity, t, logits, y)

            if trial is not None and report_intermediate:
                trial.report(
                    self.metrics_plugin.evaluator.intermediate_result(t), step=t
                )
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        # Save results to run directory
        with open(self.config.run_dir / "avalanche_results.pkl", "wb") as f:
            pickle.dump(results, f)

        results = self.metrics_plugin.evaluator.result()
        pickle.dump(results, open(self.config.run_dir / "results.pkl", "wb"))

        for key, value in asdict(results).items():
            if isinstance(value, (float, int)):
                logger.info(f"{key}: {value:.4f}")
        return results

    def count_parameters(self):
        print(parameter_summary_str(self.model))
