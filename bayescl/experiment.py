import os

import matplotlib

from bayescl.methods.clora import CLoRAAdapterFactory, CLoRAPlugin
from bayescl.methods.sdlora import SDLoRAAdapterFactory, SDLoRAPlugin
from bayescl.methods.tball import TBALLAdapterFactory
from bayescl.vbnn import replace_head

matplotlib.use("Agg")

import json
import pickle
import random
import time
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from avalanche.evaluation.metrics import (
    StreamConfusionMatrix,
    accuracy_metrics,
    forgetting_metrics,
    loss_metrics,
    timing_metrics,
)
from avalanche.logging import BaseLogger, InteractiveLogger, TensorboardLogger
from avalanche.training import DER, GDumb, Naive, ReservoirSamplingBuffer
from avalanche.training.plugins import (
    EvaluationPlugin,
    EWCPlugin,
    ReplayPlugin,
    RWalkPlugin,
    SupervisedPlugin,
    SynapticIntelligencePlugin,
)
from avalanche.training.templates import SupervisedTemplate
from loguru import logger
from optuna import Trial
from setproctitle import setproctitle
from torch import BoolTensor

from bayescl import config
from bayescl.benchmark import get_benchmark
from bayescl.methods.ball import BALLAdapterFactory
from bayescl.methods.inflora import InfLoRAAdapterFactory, InfLoRAPlugin
from bayescl.methods.lora import LoRAAdapterFactory
from bayescl.methods.train_mask import TrainTaskMask
from bayescl.methods.vcl import VCLStrategy
from bayescl.metrics.ece import (
    ExpectedCalibrationError,
)
from bayescl.metrics.plugin import MetricsPlugin
from bayescl.model import get_model
from bayescl.peft import (
    RegexFilter,
    add_adapters,
    parameter_summary_str,
)
from bayescl.plugins import ZeusMonitorPlugin


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

    def _new_log_dir(self) -> Path:
        id_ = time.strftime("%Y-%m-%d_%H-%M-%S")
        # If running with slurm use the job id
        if "SLURM_JOB_ID" in os.environ:
            id_ = f"{os.environ['SLURM_JOB_ID']}_{os.environ.get('SLURM_ARRAY_TASK_ID', 0)}"

        log_dir = (
            Path(self.cfg.log_root)
            / self.cfg.label.study
            / self.cfg.label.scenario
            / self.cfg.label.method
            / (self.cfg.label.run or id_)
        )
        log_dir.mkdir(parents=True, exist_ok=False)
        setproctitle(
            "bayescl"
            f".{self.cfg.label.study}"
            f".{self.cfg.label.scenario}"
            f".{self.cfg.label.method}"
        )
        with open(log_dir / "config.json", "w") as f:
            json.dump(self.cfg.model_dump(mode="json"), f, indent=2)

        logger.info(f"Logging to '{log_dir}'")
        return log_dir

    def _new_logger(self) -> TensorboardLogger:
        tb_logger = TensorboardLogger(self.log_dir)
        self.loggers.append(InteractiveLogger())
        self.loggers.append(tb_logger)
        return tb_logger

    def _build_peft(self):
        peft = self.cfg.peft
        if peft is None:
            return
        model_config = self.cfg.model
        if not isinstance(model_config, config.HuggingFaceModelConfig):
            raise ValueError("PEFT is only supported for HuggingFace models.")

        regex_filter = RegexFilter(model_config.adapter_filter)
        # Make recreating the random projections in T-BALL easy.
        torch.manual_seed(self.cfg.seed + 7808)

        match peft.type:
            case "LoRA":
                logger.info("Adding LoRA adapters")
                add_adapters(self.model, regex_filter, LoRAAdapterFactory(peft))
            case "BALL":
                logger.info("Adding BALL adapters")
                add_adapters(self.model, regex_filter, BALLAdapterFactory(peft))
                if peft.bll:
                    replace_head(self.model, model_config.head_module, config=peft.vbnn)
            case "SDLoRA":
                logger.info("Adding SD-LoRA adapters")
                factory = SDLoRAAdapterFactory(self.num_tasks, peft)
                add_adapters(self.model, regex_filter, factory)
                self.plugins.append(SDLoRAPlugin())
            case "TBALL":
                logger.info("Adding TBALL adapters")
                add_adapters(self.model, regex_filter, TBALLAdapterFactory(peft))
            case "CLoRA":
                logger.info("Adding CLoRA adapters and plugin")
                factory = CLoRAAdapterFactory(self.num_tasks, peft)
                add_adapters(self.model, regex_filter, factory)
                self.plugins.append(CLoRAPlugin(peft, self.tb_log.writer))
            case "InfLoRA":
                logger.info("Adding InfLoRA adapters and plugin")
                add_adapters(self.model, regex_filter, InfLoRAAdapterFactory(peft))
                self.plugins.append(
                    InfLoRAPlugin(
                        peft,
                        total_tasks=self.num_tasks,
                        optimizer_factory=self._new_optimizer,
                        max_activation_batches=peft.max_activation_batches,
                    )
                )
            case _:
                raise ValueError(f"Unsupported PEFT method: {peft.type}")

        self.model.get_submodule(model_config.head_module).requires_grad_(True)

    def _build_plugins(self):
        if self.cfg.use_local_ce:
            logger.info("Add 'TrainTaskMask' plugin")
            self.plugins.append(TrainTaskMask(self.mask, self._new_optimizer))
        if self.cfg.rwalk:
            logger.info("Add 'RWalk' plugin")
            config = self.cfg.rwalk
            self.plugins.append(
                RWalkPlugin(
                    ewc_lambda=config.ewc_lambda,
                    ewc_alpha=config.ewc_alpha,
                    delta_t=config.delta_t,
                )
            )
        if self.cfg.replay > 0:
            logger.info(f"Add replay plugin with memory size: {self.cfg.replay}")
            self.plugins.append(
                ReplayPlugin(
                    mem_size=self.cfg.replay,
                    storage_policy=ReservoirSamplingBuffer(self.cfg.replay),
                )
            )
        if self.cfg.ewc:
            logger.info("Add 'EWCPlugin' plugin")
            self.plugins.append(EWCPlugin(**self.cfg.ewc.kwargs(), mode="online"))
        if self.cfg.si:
            logger.info("Add 'SynapticIntelligencePlugin' plugin")

            # Exclude all parameters except adapters from SI regularization
            excluded_parameters = [
                name
                for name, param in self.model.named_parameters()
                if not param.requires_grad
            ]
            self.plugins.append(
                SynapticIntelligencePlugin(
                    **self.cfg.si.kwargs(), excluded_parameters=excluded_parameters
                )
            )
        if self.cfg.zeus_monitor:
            logger.info("Add 'ZeusMonitorPlugin' plugin")
            self.zeus_monitor_plugin = ZeusMonitorPlugin(
                config=self.cfg.zeus_monitor,
                writer=self.tb_log.writer,  # tpye: ignore
            )
            self.plugins.append(self.zeus_monitor_plugin)

        self.plugins.append(self.metrics_plugin)

    def _preflight(self):
        logger.info("Resolved Config:\n{}", pformat(self.cfg.model_dump(mode="python")))
        logger.info("Parameter Counts:\n{}", parameter_summary_str(self.model))
        logger.info("Plugins:\n{}", [type(p).__name__ for p in self.plugins])

    def _seed_everything(self):
        if self.cfg.seed is not None:
            logger.info(f"Set random seed to {self.cfg.seed}")
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
            random.seed(self.cfg.seed)

    def __init__(self, cfg: config.Config) -> None:
        self.cfg = cfg
        self._seed_everything()
        self.plugins: List[SupervisedPlugin] = []
        self.loggers: List[BaseLogger] = []

        self.benchmark = get_benchmark(cfg)
        self.mask = class_schedule_to_task_mask(
            avalanche_class_schedule(self.benchmark), self.benchmark.n_classes
        )
        self.num_tasks: int = len(self.benchmark.train_stream)
        self.num_classes: int = self.benchmark.n_classes
        self.log_dir: Path = self._new_log_dir()
        self.tb_log = self._new_logger()
        self.eval_plugin: EvaluationPlugin = self._new_eval_plugin()
        self.metrics_plugin = MetricsPlugin(self.num_tasks, self.num_classes)
        self.zeus_monitor_plugin: ZeusMonitorPlugin | None = None
        self.model = get_model(cfg, self.benchmark.n_classes)
        self._build_peft()
        self._build_plugins()

    def _new_optimizer(self, parameters) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, parameters), lr=self.cfg.lr
        )

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

    def _build_strategy(self) -> SupervisedTemplate:
        strategy = self.cfg.strategy
        kwargs = dict(
            model=self.model,
            optimizer=self._new_optimizer(self.model.parameters()),
            train_mb_size=self.cfg.train_mb_size,
            eval_mb_size=self.cfg.eval_mb_size or self.cfg.train_mb_size,
            train_epochs=self.cfg.epochs,
            evaluator=self.eval_plugin,
            device=self.cfg.device,
            plugins=self.plugins,
            eval_every=self.cfg.eval_every,
            criterion=torch.nn.CrossEntropyLoss(),
        )
        match strategy.type:
            case "Naive":
                return Naive(**kwargs)  # type: ignore
            case "VCL":
                assert isinstance(strategy, config.VCLConfig)
                logger.info("Using Variational Continual Learning (VCL) strategy")
                return VCLStrategy(
                    config=strategy,
                    mask=self.mask,
                    writer=self.tb_log.writer,
                    optimizer_fn=self._new_optimizer,
                    **kwargs,
                )
            case "DER":
                logger.info("Using Dark Experience Replay (DER) strategy")
                return DER(**kwargs, **strategy.kwargs())  # type: ignore
            case "GDumb":
                logger.info("Using GDumb strategy")
                return GDumb(**kwargs, **strategy.kwargs())  # type: ignore
            case _:
                raise ValueError(f"Unsupported strategy: {strategy.type}")

    def run(self, trial: Trial | None = None) -> tuple[float, float]:
        self._preflight()
        strategy = self._build_strategy()
        strategy.mask = self.mask.to(self.cfg.device)  # type: ignore

        # TRAINING LOOP
        logger.info("Starting experiment...")
        results: Sequence[Dict[str, float]] = []
        for t, experience in enumerate(self.benchmark.train_stream):
            logger.info(f"Start of experience: {experience.current_experience}")
            logger.info(f"Experience Size: {len(experience.dataset)}")
            logger.info(f"Current Classes: {experience.classes_in_this_experience}")

            # If first_exp_epochs is set, use it for the first experience
            strategy.train_epochs = (
                self.cfg.first_exp_epochs
                if t == 0 and self.cfg.first_exp_epochs is not None
                else self.cfg.epochs
            )

            # train returns a dictionary which contains all the metric values
            strategy.train(
                experience,
                self.benchmark.test_stream[: t + 1],
                num_workers=self.cfg.num_workers,
            )

            results.append(
                strategy.eval(
                    self.benchmark.test_stream, num_workers=self.cfg.num_workers
                )
            )
            if self.cfg.checkpoint:
                checkpoint_path = self.log_dir / f"checkpoint-t{t:02d}.pth"
                self.save_checkpoint(checkpoint_path)

            if self.cfg.max_tasks is not None and t + 1 >= self.cfg.max_tasks:
                logger.info(f"Stopping after {self.cfg.max_tasks} tasks")
                break

        # Save results to log directory
        with open(self.log_dir / "avalanche_results.pkl", "wb") as f:
            pickle.dump(results, f)

        if self.zeus_monitor_plugin is not None:
            with open(self.log_dir / "zeus_monitor.json", "w") as f:
                json.dump(self.zeus_monitor_plugin.result(), f, indent=2)

        if self.metrics_plugin is not None:
            metrics, raw_data = self.metrics_plugin.evaluator.result()
            pickle.dump(metrics, open(self.log_dir / "metrics.pkl", "wb"))
            pickle.dump(raw_data, open(self.log_dir / "raw_data.pkl", "wb"))

            for key, value in metrics.items():
                if isinstance(value, (float, int)):
                    logger.info(f"{key}: {value:.4f}")
            return metrics["accuracy_seen_avg"], metrics["ece_seen_avg"]
        return -1, -1

    def count_parameters(self):
        print(parameter_summary_str(self.model))
