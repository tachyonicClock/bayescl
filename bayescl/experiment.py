import pickle
import time
from pathlib import Path
from pprint import pprint
from typing import List

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
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from claiutil.peft import (
    BLoB,
    LoRA_Factory,
    add_adapters,
    print_parameter_summary,
    set_module,
)
from claiutil.vbnn import VariationalLinear
from loguru import logger

from bayescl import config
from bayescl.benchmark import get_benchmark
from bayescl.model import get_model, get_peft_filter
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
        elif peft_cfg.type == "BLoB":
            # Add plugin that contributes kl divergence loss
            self.plugins += [VBNNPlugin(peft_cfg.beta, self.tb_log.writer)]

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
                config=peft_cfg.config.blob_A,
            )  # type: ignore
            set_module(self.model, peft_cfg.head_module, new_linear)

    def _preflight(self):
        pprint(self.cfg.model_dump(mode="python"))
        print_parameter_summary(self.model)

    def __init__(self, cfg: config.Config) -> None:
        self.cfg = cfg
        self.plugins: List[SupervisedPlugin] = []
        self.loggers: List[BaseLogger] = []

        self.benchmark = get_benchmark(cfg)
        self.log_dir: Path = self._new_log_dir()
        self.tb_log = self._new_logger()
        self.eval_plugin: EvaluationPlugin = self._new_eval_plugin()
        self.model = get_model(cfg, self.benchmark.n_classes)
        self._add_peft_adapters()

    def run(self):
        self._preflight()
        strategy = Naive(
            model=self.model,
            optimizer=torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr),
            criterion=torch.nn.CrossEntropyLoss(),
            train_mb_size=self.cfg.train_mb_size,
            eval_mb_size=self.cfg.eval_mb_size or self.cfg.train_mb_size,
            train_epochs=self.cfg.train_epochs,
            evaluator=self.eval_plugin,
            device=self.cfg.device,
            plugins=self.plugins,
        )

        # TRAINING LOOP
        logger.info("Starting experiment...")
        results = []
        for experience in self.benchmark.train_stream:
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

        # Save results to log directory
        with open(self.log_dir / "results.pkl", "wb") as f:
            pickle.dump(results, f)
