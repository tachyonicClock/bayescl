import os

import matplotlib

from bayescl.plugins.clora import CLoRAPlugin
from bayescl.plugins.inflora import PluginInfLoRA

matplotlib.use("Agg")

import pickle
import time
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Sequence

import torch
import yaml
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
    ReplayPlugin,
    RWalkPlugin,
    SupervisedPlugin,
)
from avalanche.training.templates import SupervisedTemplate
from loguru import logger
from optuna import Trial
from setproctitle import setproctitle
from torch import BoolTensor

from bayescl import config
from bayescl.benchmark import get_benchmark
from bayescl.metrics.ece import (
    ExpectedCalibrationError,
)
from bayescl.metrics.plugin import MetricsPlugin
from bayescl.model import get_model, get_peft_filter
from bayescl.peft import (
    BALL,
    CLoRA,
    CLoRAConfig,
    LoRA_Factory,
    RegexFilter,
    add_adapters,
    iter_adapter_parameters,
    iter_named_adapters,
    parameter_summary_str,
    set_module,
)
from bayescl.plugins.ball import BALLPlugin
from bayescl.plugins.train_mask import TrainTaskMask
from bayescl.vbnn import VariationalLinear


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

        if self.cfg.run_id is not None:
            id_ = f"{self.cfg.run_id}"

        log_dir = (
            Path(self.cfg.log_root)
            / self.cfg.study_name
            / f"{self.cfg.scenario.dataset}"
            / self.cfg.label
            / id_
        )
        log_dir.mkdir(parents=True, exist_ok=False)

        with open(log_dir / "config.yaml", "w") as f:
            yaml.dump(self.cfg.model_dump(mode="json"), f)

        return log_dir

    def _new_logger(self) -> TensorboardLogger:
        tb_logger = TensorboardLogger(self.log_dir)
        self.loggers.append(InteractiveLogger())
        self.loggers.append(tb_logger)
        return tb_logger

    def _add_peft_adapters(self):
        peft = self.cfg.peft
        if not peft:
            return
        filter_regex = RegexFilter(self.cfg.model.adapter_filter)

        logger.info(f"Add PEFT adapter: {peft}")
        if peft.type == "LoRA":
            add_adapters(
                self.model,
                get_peft_filter(self.cfg),
                LoRA_Factory(
                    r=peft.r,
                    lora_alpha=peft.lora_alpha,
                    lora_dropout=peft.lora_dropout,
                ),
            )
            self.model.get_submodule(peft.head_module).requires_grad_(True)
        elif peft.type == "CLoRA" or peft.type == "InfLoRA":
            add_adapters(self.model, filter_regex, CLoRA(CLoRAConfig(r=peft.r)))

            if peft.type == "InfLoRA":
                self.plugins.append(
                    PluginInfLoRA(threshold=peft.threshold, total_tasks=self.num_tasks)
                )
            elif peft.type == "CLoRA":
                self.plugins.append(CLoRAPlugin(peft.lambda_, self.tb_log.writer))
            self.model.get_submodule(peft.head_module).requires_grad_(True)
        elif peft.type == "BALL":
            # Add plugin that contributes kl divergence loss
            self.plugins += [
                BALLPlugin(
                    beta=peft.beta,
                    bayes_eval_samples=peft.bayes_eval_samples,
                    writer=self.tb_log.writer,
                )
            ]

            # Add BALL adapters
            factory = BALL(peft.config)
            add_adapters(self.model, filter_regex, factory)

            if peft.vbll:
                logger.info("Using variational bayesian last layer (VBLL)")
                # Replace classifier with VBNN linear layer
                linear = self.model.get_submodule(peft.head_module)
                assert isinstance(linear, torch.nn.Linear)
                new_linear = VariationalLinear(
                    in_features=linear.in_features,
                    out_features=linear.out_features,
                    config=peft.config.vbnn,
                )  # type: ignore
                set_module(self.model, peft.head_module, new_linear)
            else:
                logger.info("Using standard last layer")
                self.model.get_submodule(peft.head_module).requires_grad_(True)

        # Optionally load adapter weights from checkpoint
        if peft.checkpoint is not None:
            logger.info(f"Loading checkpoint from '{peft.checkpoint}'.")
            state_dict = torch.load(peft.checkpoint, map_location="cpu")
            _, unexpected = self.model.load_state_dict(state_dict, strict=False)
            assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

        logger.info("ADAPTERS:")
        for name, _ in iter_named_adapters(self.model):
            logger.info(f"{name}")

    def _add_replay_plugin(self):
        if self.cfg.replay > 0:
            logger.info(f"Add replay plugin with memory size: {self.cfg.replay}")
            self.plugins.append(
                ReplayPlugin(
                    mem_size=self.cfg.replay,
                    storage_policy=ReservoirSamplingBuffer(self.cfg.replay),
                )
            )

    def _add_plugins(self):
        if self.cfg.use_local_ce:
            logger.info("Add 'TrainTaskMask' plugin")
            self.plugins.append(TrainTaskMask(self.mask))
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

        self.plugins.append(self.metrics_plugin)

    def _preflight(self):
        logger.info("Resolved Config:\n{}", pformat(self.cfg.model_dump(mode="python")))
        logger.info("Parameter Counts:\n{}", parameter_summary_str(self.model))
        logger.info("Plugins:\n{}", [type(p).__name__ for p in self.plugins])

    def __init__(self, cfg: config.Config) -> None:
        self.cfg = cfg
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
        self.model = get_model(cfg, self.benchmark.n_classes)
        self._add_peft_adapters()
        self._add_replay_plugin()
        self._add_plugins()

    def get_strategy(self) -> SupervisedTemplate:
        base_kwargs = dict(
            model=self.model,
            optimizer=torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr),
            criterion=torch.nn.CrossEntropyLoss(),
            train_mb_size=self.cfg.train_mb_size,
            eval_mb_size=self.cfg.eval_mb_size or self.cfg.train_mb_size,
            train_epochs=self.cfg.epochs,
            evaluator=self.eval_plugin,
            device=self.cfg.device,
            plugins=self.plugins,
        )

        strategy = self.cfg.strategy
        if strategy is None:
            return Naive(**base_kwargs)
        elif isinstance(strategy, config.DERConfig):
            logger.info(
                f"DER(mem_size={strategy.mem_size}, alpha={strategy.alpha}, beta={strategy.beta})"
            )
            return DER(
                mem_size=strategy.mem_size,
                alpha=strategy.alpha,
                beta=strategy.beta,
                **base_kwargs,
            )
        elif isinstance(strategy, config.GDumbConfig):
            logger.info(f"GDumb(mem_size={strategy.mem_size})")
            return GDumb(
                mem_size=strategy.mem_size,
                **base_kwargs,
            )
        raise ValueError(f"Unknown strategy: {strategy}")

    def run(self, trial: Trial | None = None) -> tuple[float, float]:
        setproctitle(f"bayescl.{self.cfg.label}")
        self._preflight()
        strategy = self.get_strategy()

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
            strategy.train(experience, num_workers=self.cfg.num_workers)
            logger.info("Training completed")

            logger.info("Computing accuracy on the whole test set")
            # eval also returns a dictionary which contains all the metric values
            results.append(
                strategy.eval(
                    self.benchmark.test_stream, num_workers=self.cfg.num_workers
                )
            )

            if self.cfg.max_tasks is not None and t + 1 >= self.cfg.max_tasks:
                logger.info(f"Stopping after {self.cfg.max_tasks} tasks")
                break

        # Save results to log directory
        with open(self.log_dir / "avalanche_results.pkl", "wb") as f:
            pickle.dump(results, f)

        metrics = self.metrics_plugin.evaluator.result()
        pickle.dump(metrics, open(self.log_dir / "metrics.pkl", "wb"))

        # Optionally save adapter weights
        if self.cfg.peft and self.cfg.peft.save:
            filename = self.log_dir / "adapter.pth"
            logger.info(f"Saving adapter weights to '{filename}'.")
            with open(filename, "wb") as f:
                state = dict(iter_adapter_parameters(self.model))
                head = self.cfg.peft.head_module
                state.update(
                    self.model.get_submodule(head).state_dict(prefix=head + ".")
                )
                torch.save(state, f)

        return metrics["accuracy_seen_avg"], metrics["ece_final"]
