import matplotlib

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
from claiutil.avalanche import AvalancheResults
from claiutil.peft import (
    BLoB,
    CLoRA,
    CLoRAConfig,
    LoRA_Factory,
    add_adapters,
    iter_named_adapters,
    parameter_summary_str,
    set_module,
)
from claiutil.vbnn import VariationalLinear
from loguru import logger
from setproctitle import setproctitle

from bayescl import config
from bayescl.benchmark import get_benchmark
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
            loggers=self.loggers,
        )

    def _new_log_dir(self) -> Path:
        log_dir = (
            Path(self.cfg.log_root)
            / self.cfg.study_name
            / f"{self.cfg.scenario.dataset}"
            / self.cfg.label
            / time.strftime("%Y-%m-%d_%H-%M-%S")
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
        peft_cfg = self.cfg.peft
        if not peft_cfg:
            return
        logger.info(f"Add PEFT adapter: {peft_cfg}")

        if peft_cfg.type == "LoRA":
            add_adapters(
                self.model,
                get_peft_filter(self.cfg),
                LoRA_Factory(
                    r=peft_cfg.r,
                    lora_alpha=peft_cfg.lora_alpha,
                    lora_dropout=peft_cfg.lora_dropout,
                ),
            )
            self.model.get_submodule(peft_cfg.head_module).requires_grad_(True)
        elif peft_cfg.type == "CLoRA" or peft_cfg.type == "InfLoRA":
            add_adapters(
                self.model,
                get_peft_filter(self.cfg),
                CLoRA(CLoRAConfig(r=peft_cfg.r)),
            )
            if peft_cfg.type == "InfLoRA":
                self.plugins.append(
                    PluginInfLoRA(
                        threshold=peft_cfg.threshold,
                        total_tasks=self.num_tasks,
                    )
                )
            elif peft_cfg.type == "CLoRA":
                self.plugins.append(CLoRAPlugin(peft_cfg.beta, self.tb_log.writer))
            self.model.get_submodule(peft_cfg.head_module).requires_grad_(True)
        elif peft_cfg.type == "BLoB":
            # Add plugin that contributes kl divergence loss
            self.plugins += [
                VBNNPlugin(
                    beta=peft_cfg.beta,
                    bayes_eval_samples=peft_cfg.bayes_eval_samples,
                    writer=self.tb_log.writer,
                )
            ]

            # Add BLoB adapters
            factory = BLoB(peft_cfg.config)
            add_adapters(self.model, get_peft_filter(self.cfg), factory)

            # Replace classifier with VBNN linear layer
            linear = self.model.get_submodule(peft_cfg.head_module)
            assert isinstance(linear, torch.nn.Linear)
            new_linear = VariationalLinear(
                dim_in=linear.in_features,
                dim_out=linear.out_features,
                bias=linear.bias is not None,
                config=peft_cfg.config.blob_B,
            )  # type: ignore
            set_module(self.model, peft_cfg.head_module, new_linear)

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
            logger.info("Add `TrainTaskMask` plugin")
            self.plugins.append(TrainTaskMask(self.benchmark))
        self.plugins.append(self.capymoa_ocl_metrics)

    def _preflight(self):
        logger.info("Resolved Config:\n{}", pformat(self.cfg.model_dump(mode="python")))
        logger.info("Parameter Counts:\n{}", parameter_summary_str(self.model))
        logger.info("Plugins:\n{}", [type(p).__name__ for p in self.plugins])

    def __init__(self, cfg: config.Config) -> None:
        self.cfg = cfg
        self.plugins: List[SupervisedPlugin] = []
        self.loggers: List[BaseLogger] = []

        self.benchmark = get_benchmark(cfg)
        self.num_tasks: int = len(self.benchmark.train_stream)
        self.num_classes: int = self.benchmark.n_classes
        self.log_dir: Path = self._new_log_dir()
        self.tb_log = self._new_logger()
        self.eval_plugin: EvaluationPlugin = self._new_eval_plugin()
        self.capymoa_ocl_metrics = AvalancheResults(self.num_tasks, self.num_classes)
        self.model = get_model(cfg, self.benchmark.n_classes)
        self._add_peft_adapters()
        self._add_replay_plugin()
        self._add_plugins()

    def run(self) -> float:
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
            logger.info(f"Current Classes: {experience.classes_in_this_experience}")

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

        metrics = self.capymoa_ocl_metrics.build()
        metrics.save(self.log_dir / "results")

        return metrics.accuracy_seen_avg
