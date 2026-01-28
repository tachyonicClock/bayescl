import os

import matplotlib

matplotlib.use("Agg")

import pickle
import random
import time
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Sequence

import numpy as np
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

import bayescl.methods.l2p as l2p
from bayescl import config
from bayescl.benchmark import get_benchmark
from bayescl.methods.ball import BALLStrategy
from bayescl.methods.l2p import Backbone
from bayescl.methods.train_mask import TrainTaskMask
from bayescl.metrics.ece import (
    ExpectedCalibrationError,
)
from bayescl.metrics.plugin import MetricsPlugin
from bayescl.model import get_model
from bayescl.peft import (
    BALL,
    LoRA_Factory,
    RegexFilter,
    add_adapters,
    iter_named_adapters,
    parameter_summary_str,
    set_module,
)
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
        with open(log_dir / "config.yaml", "w") as f:
            yaml.dump(self.cfg.model_dump(mode="json"), f)

        logger.info(f"Logging to '{log_dir}'")
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
                filter_regex,
                LoRA_Factory(
                    r=peft.r,
                    lora_alpha=peft.lora_alpha,
                    lora_dropout=peft.lora_dropout,
                ),
            )
            self.model.get_submodule(peft.head_module).requires_grad_(True)
        elif isinstance(peft, config.L2PConfig):
            assert isinstance(self.model, Backbone)
            logger.info("Using L2P prompt-based method")
            del self.model.model.classifier  # type: ignore
            self.plugins += [l2p.L2PPlugin()]
            self.model = l2p.L2PModel(
                backbone=self.model,
                num_classes=self.benchmark.n_classes,
                pull_constraint_coeff=peft.pull_constraint_coeff,
                prompt_pool=l2p.PromptPool(
                    pool_size=peft.pool_size,
                    prompt_length=peft.prompt_length,
                    embed_dim=self.model.embed_dim,
                    top_k=peft.top_k,
                ),
            )
        elif peft.type == "BALL":
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

    def _add_plugins(self):
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
        self.model = get_model(cfg, self.benchmark.n_classes)
        self._add_peft_adapters()
        self._add_plugins()

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

    def get_strategy(self) -> SupervisedTemplate:
        base_kwargs = dict(
            model=self.model,
            optimizer=self._new_optimizer(self.model.parameters()),
            train_mb_size=self.cfg.train_mb_size,
            eval_mb_size=self.cfg.eval_mb_size or self.cfg.train_mb_size,
            train_epochs=self.cfg.epochs,
            evaluator=self.eval_plugin,
            device=self.cfg.device,
            plugins=self.plugins,
            eval_every=self.cfg.eval_every,
        )

        strategy = self.cfg.strategy
        if self.cfg.peft and self.cfg.peft.type == "BALL":
            assert strategy is None, "BALL sets its own strategy"
            return BALLStrategy(
                beta=self.cfg.peft.beta,
                train_samples=self.cfg.peft.train_samples,
                test_samples=self.cfg.peft.test_samples,
                writer=self.tb_log.writer,
                mask=self.mask,
                optimizer_fn=self._new_optimizer,
                first_task_beta=self.cfg.peft.first_task_beta,
                softmax_avg=self.cfg.peft.softmax_avg,
                **base_kwargs,
            )
        elif strategy is None:
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
                criterion=torch.nn.CrossEntropyLoss(),
                **base_kwargs,
            )
        raise ValueError(f"Unknown strategy: {strategy}")

    def run(self, trial: Trial | None = None) -> tuple[float, float]:
        self._preflight()
        strategy = self.get_strategy()
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
                experience, self.benchmark.test_stream, num_workers=self.cfg.num_workers
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

        metrics, raw_data = self.metrics_plugin.evaluator.result()
        pickle.dump(metrics, open(self.log_dir / "metrics.pkl", "wb"))
        pickle.dump(raw_data, open(self.log_dir / "raw_data.pkl", "wb"))

        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                logger.info(f"{key}: {value:.4f}")
        return metrics["accuracy_seen_avg"], metrics["ece_seen_avg"]
