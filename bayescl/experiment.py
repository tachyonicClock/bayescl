import matplotlib

matplotlib.use("Agg")

import json
import pickle
import random
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
from bayescl.metrics.ece import (
    ExpectedCalibrationError,
)
from bayescl.metrics.plugin import MetricsPlugin
from bayescl.model import get_model
from bayescl.peft import parameter_summary_str


def build_experiment(
    arm,
    *,
    dataset,
    scale,
    seed: int,
    validation: bool,
    run_dir: Path,
    dataset_root: Path,
    device: str = "cuda",
) -> "Experiment":
    backbone = dataset.backbone
    return arm.build(
        Experiment(
            arm=arm,
            dataset=dataset.scenario,
            n_tasks=dataset.n_tasks,
            shuffle=dataset.shuffle,
            dataset_root=Path(dataset_root),
            standardize=dataset.standardize,
            validation=validation,
            backbone_name=backbone.name,
            freeze_backbone=backbone.freeze_backbone,
            adapter_filter=backbone.adapter_filter,
            head_module=backbone.head_module,
            epochs=scale.epochs(dataset),
            train_mb_size=dataset.train_mb_size,
            eval_mb_size=dataset.eval_mb_size,
            num_workers=dataset.num_workers,
            seed=seed,
            run_dir=run_dir,
            device=device,
        )
    )


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
        run_dir = self.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        setproctitle(f"bayescl.{self.dataset}")
        with open(run_dir / "spec.json", "w") as f:
            json.dump(self._config(), f, indent=2, default=str)
        logger.info(f"Logging to '{run_dir}'")
        return run_dir

    def _new_logger(self) -> TensorboardLogger:
        tb_logger = TensorboardLogger(self.run_dir)
        self.loggers.append(InteractiveLogger())
        self.loggers.append(tb_logger)
        return tb_logger

    def _preflight(self):
        logger.info("Resolved Spec:\n{}", pformat(self._config()))
        logger.info("Parameter Counts:\n{}", parameter_summary_str(self.model))
        logger.info("Plugins:\n{}", [type(p).__name__ for p in self.plugins])

    def _seed_everything(self):
        if self.seed is not None:
            logger.info(f"Set random seed to {self.seed}")
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

    def __init__(self, *, arm, **config) -> None:
        config.setdefault("first_exp_epochs", None)
        config.setdefault("eval_every", -1)
        config.setdefault("checkpoint", False)
        self.__dict__.update(config)
        self.arm = arm
        self._seed_everything()
        self.plugins: List[SupervisedPlugin] = []
        self.loggers: List[BaseLogger] = []

        self.benchmark = get_benchmark(self)
        self.mask = class_schedule_to_task_mask(
            avalanche_class_schedule(self.benchmark), self.benchmark.n_classes
        )
        self.num_tasks: int = len(self.benchmark.train_stream)
        self.num_classes: int = self.benchmark.n_classes
        self.run_dir: Path = self._prepare_run_dir()
        self.tb_log = self._new_logger()
        self.eval_plugin: EvaluationPlugin = self._new_eval_plugin()
        self.metrics_plugin = MetricsPlugin(self.num_tasks, self.num_classes)
        self.model = get_model(self, self.benchmark.n_classes)
        self.arm._build_peft(self)
        self.arm._build_plugins(self)

    def save_checkpoint(self, filename: Path) -> None:
        # Only save learnable parameters (adapters)
        state = {
            k: v.to(copy=True, dtype=torch.float16)
            for k, v in self.model.named_parameters()
            if v.requires_grad
        }
        numel = sum(p.numel() for p in state.values())
        logger.info(f"Saving checkpoint to '{filename}' ({numel} parameters)")
        torch.save(state, filename)

    def _config(self) -> dict:
        return {
            name: getattr(self, name)
            for name in (
                "dataset", "n_tasks", "shuffle", "dataset_root", "standardize",
                "validation", "backbone_name", "freeze_backbone", "adapter_filter",
                "head_module", "epochs", "train_mb_size", "eval_mb_size",
                "num_workers", "run_dir", "seed", "device", "first_exp_epochs",
                "eval_every", "checkpoint",
            )
        }

    def run(
        self, trial: Trial | None = None, *, report_intermediate: bool = True
    ) -> tuple[float, float]:
        self._preflight()
        strategy = self.arm._build_strategy(self)
        strategy.mask = self.mask.to(self.device)  # type: ignore

        # TRAINING LOOP
        logger.info("Starting experiment...")
        results: Sequence[Dict[str, float]] = []
        for t, experience in enumerate(self.benchmark.train_stream):
            logger.info(f"Start of experience: {experience.current_experience}")
            logger.info(f"Experience Size: {len(experience.dataset)}")
            logger.info(f"Current Classes: {experience.classes_in_this_experience}")

            # If first_exp_epochs is set, use it for the first experience
            strategy.train_epochs = (
                self.first_exp_epochs
                if t == 0 and self.first_exp_epochs is not None
                else self.epochs
            )

            # train returns a dictionary which contains all the metric values
            strategy.train(
                experience,
                self.benchmark.test_stream[: t + 1],
                num_workers=self.num_workers,
            )

            results.append(
                strategy.eval(
                    self.benchmark.test_stream, num_workers=self.num_workers
                )
            )
            if self.checkpoint:
                checkpoint_path = self.run_dir / f"checkpoint-t{t:02d}.pth"
                self.save_checkpoint(checkpoint_path)

            if trial is not None and report_intermediate:
                intermediate_acc, intermediate_ece = (
                    self.metrics_plugin.evaluator.intermediate_result(t)
                )
                intermediate_score = 0.5 * (intermediate_acc + (1 - intermediate_ece))
                trial.report(intermediate_score, step=t)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        # Save results to run directory
        with open(self.run_dir / "avalanche_results.pkl", "wb") as f:
            pickle.dump(results, f)

        metrics, raw_data = self.metrics_plugin.evaluator.result()
        pickle.dump(metrics, open(self.run_dir / "metrics.pkl", "wb"))
        pickle.dump(raw_data, open(self.run_dir / "raw_data.pkl", "wb"))

        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                logger.info(f"{key}: {value:.4f}")
        return metrics["accuracy_seen_avg"], metrics["ece_seen_avg"]

    def count_parameters(self):
        print(parameter_summary_str(self.model))
