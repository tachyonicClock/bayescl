import os

import matplotlib

from bayescl.datasets import TinyImageNet
from bayescl.plugins.clora import CLoRAPlugin
from bayescl.plugins.inflora import PluginInfLoRA

matplotlib.use("Agg")

import pickle
import time
from pathlib import Path
from pprint import pformat
from typing import Dict, List, Sequence

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
from avalanche.training import Naive, ReservoirSamplingBuffer
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin, SupervisedPlugin
from claiutil.avalanche import (
    ExpectedCalibrationError,
    OutlierExposure,
    TaskIncrementalAccuracy,
)
from claiutil.datasets import avalanche_class_schedule, class_schedule_to_task_mask
from claiutil.peft import (
    BLoB,
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
from claiutil.vbnn import VariationalLinear
from loguru import logger
from optuna import Trial
from optuna.exceptions import TrialPruned
from setproctitle import setproctitle

from bayescl import config
from bayescl.benchmark import get_benchmark, get_transforms
from bayescl.metrics.plugin import MetricsPlugin
from bayescl.model import get_model, get_peft_filter
from bayescl.plugins.train_mask import TrainTaskMask
from bayescl.plugins.vbnn import VBNNPlugin


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
            TaskIncrementalAccuracy(self.mask),
            loggers=self.loggers,
        )

    def _new_log_dir(self) -> Path:
        id_ = time.strftime("%Y-%m-%d_%H-%M-%S")
        # If running with slurm use the job id
        if "SLURM_JOB_ID" in os.environ:
            id_ = f"{os.environ['SLURM_JOB_ID']}_{os.environ.get('SLURM_ARRAY_TASK_ID', 0)}"

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
        elif peft.type == "BLoB":
            # Add plugin that contributes kl divergence loss
            self.plugins += [
                VBNNPlugin(
                    beta=peft.beta,
                    bayes_eval_samples=peft.bayes_eval_samples,
                    writer=self.tb_log.writer,
                )
            ]

            # Add BLoB adapters
            factory = BLoB(peft.config)
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
        if self.cfg.replay <= 0:
            return
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

        if self.cfg.outlier_exposure:
            logger.info("Add 'OutlierExposure' plugin")
            config = self.cfg.outlier_exposure

            transform, _ = get_transforms(self.cfg, "TinyImageNet")
            outlier_dataset = TinyImageNet(
                self.cfg.dataset_root, transform, split="train"
            )
            plugin = OutlierExposure(
                torch.utils.data.DataLoader(
                    outlier_dataset,
                    batch_size=config.batch_size or self.cfg.train_mb_size,
                    shuffle=True,
                    num_workers=self.cfg.num_workers,
                ),
                strength=config.strength,
                mask=self.mask if self.cfg.use_local_ce else None,
                logger=self.tb_log,
            )
            self.plugins.append(plugin)

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

    def run(self, trial: Trial | None = None) -> tuple[float, float]:
        setproctitle(f"bayescl.{self.cfg.label}")
        self._preflight()
        strategy = Naive(
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
            accuracy = results[-1][
                f"Top1_Acc_Stream/eval_phase/test_stream/Task{self.num_tasks - 1:03d}"
            ]
            if trial is not None:
                trial.report(accuracy, t)
            if trial is not None and trial.should_prune():
                logger.warning("Trial was pruned")
                raise TrialPruned()

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
