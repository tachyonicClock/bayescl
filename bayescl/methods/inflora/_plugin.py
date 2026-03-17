from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from avalanche.training.plugins import SupervisedPlugin
from loguru import logger
from torch import nn

from ._config import InfLoRAConfig
from ._module import InfLoRAEngine


@dataclass
class ExperienceContext:
    task_id: str
    task_index: int


class InfLoRAPlugin(SupervisedPlugin):
    """Avalanche plugin for InfLoRA training."""

    supports_distributed = False

    def __init__(
        self,
        config: InfLoRAConfig,
        total_tasks: int,
        optimizer_factory,
        max_activation_batches: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.total_tasks = total_tasks
        self.optimizer_factory = optimizer_factory
        self.max_activation_batches = max_activation_batches

        self._initialized = False
        self._engine: InfLoRAEngine | None = None
        self._current_exp: ExperienceContext | None = None

    def before_training(self, strategy: Any, *args, **kwargs) -> None:
        """Initialize the InfLoRA engine before training."""
        if self._initialized:
            return
        self._engine = InfLoRAEngine(
            strategy.model,
            rank=self.config.rank,
            threshold_start=self.config.threshold_start,
            threshold_end=self.config.threshold_end,
            total_tasks=self.total_tasks,
        )
        self._initialized = True

    def before_training_exp(self, strategy: Any, *args, **kwargs) -> None:
        """Begin task before training experience."""
        engine = self._require_engine()
        context = self._experience_context(strategy)
        loader = self._activation_loader(strategy)

        logger.info(f"InfLoRA begin task {context.task_index}")
        engine.begin_task(
            task_id=context.task_id,
            task_index=context.task_index,
            data_loader=loader,
            device=strategy.device,
            max_batches=self.max_activation_batches,
        )

        self._current_exp = context
        self._configure_optimizer(strategy)

    def after_training_exp(self, strategy: Any, *args, **kwargs) -> None:
        """Finalize task after training experience."""
        engine = self._require_engine()
        context = self._current_exp or self._experience_context(strategy)
        loader = self._activation_loader(strategy)

        logger.info(f"InfLoRA finalize task {context.task_index}")
        engine.finalize_task(
            task_id=context.task_id,
            task_index=context.task_index,
            data_loader=loader,
            device=strategy.device,
            max_batches=self.max_activation_batches,
        )
        self._current_exp = None

    def _require_engine(self) -> InfLoRAEngine:
        """Get initialized engine or raise error."""
        if self._engine is None:
            raise RuntimeError(
                "plugin not initialized; before_training hook should run first"
            )
        return self._engine

    def _experience_context(self, strategy: Any) -> ExperienceContext:
        """Extract experience context from strategy."""
        exp = getattr(strategy, "experience", None)
        exp_id = getattr(exp, "current_experience", None)
        if exp_id is None:
            exp_id = getattr(getattr(strategy, "clock", None), "train_exp_counter", 0)
        task_id = str(exp_id)
        task_index = int(exp_id)
        return ExperienceContext(task_id=task_id, task_index=task_index)

    def _activation_loader(self, strategy: Any):
        """Create data loader for activation collection."""
        dataset = getattr(strategy, "adapted_dataset", None)
        if dataset is None:
            exp = getattr(strategy, "experience", None)
            dataset = getattr(exp, "dataset", None)
        if dataset is None:
            raise RuntimeError(
                "unable to access dataset from strategy for activation collection"
            )

        configured_batch_size = max(1, int(self.config.activation_batch_size))
        strategy_batch_size = int(getattr(strategy, "train_mb_size", configured_batch_size))
        batch_size = min(configured_batch_size, strategy_batch_size)
        num_workers = 0
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def _configure_optimizer(self, strategy: Any) -> None:
        """Configure optimizer with trainable InfLoRA parameters."""
        engine = self._require_engine()
        extra_params = self._select_extra_trainables(strategy.model)
        trainable = engine.trainable_parameters(extra_parameters=extra_params)
        if not trainable:
            raise RuntimeError("no trainable parameters selected for InfLoRA")
        strategy.optimizer = self.optimizer_factory(trainable)

    def _select_extra_trainables(self, model: nn.Module) -> list[nn.Parameter]:
        """Select trainable parameters outside of InfLoRA branches."""
        selected: list[nn.Parameter] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".branches." in name:
                continue
            selected.append(parameter)
        return selected
