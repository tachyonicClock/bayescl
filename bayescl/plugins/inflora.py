"""Plugin implementing InfLoRA

Liang, Y.-S., & Li, W.-J. (2024). InfLoRA: Interference-Free Low-Rank Adaptation for
Continual Learning. 23638–23647.
"""

from typing import Any, Dict, Optional, Sequence

from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import BaseSGDTemplate
from claiutil.peft import CLoRA, iter_named_adapters
from claiutil.peft._clora.layer import CLoRALayer
from claiutil.peft.inflora import (
    create_gpm,
    orthogonal_projection_matrix,
    sample_module_inputs,
    update_gpm,
)
from loguru import logger
from torch import Tensor


def threshold_schedule(task: int, total_tasks: int, initial_threshold: float) -> float:
    return initial_threshold + ((1 - initial_threshold) * task) / total_tasks


class PluginInfLoRA(SupervisedPlugin):
    def __init__(self, threshold: float, total_tasks: int):
        # Threshold (0, 1) for gradient projection memory higher is more stable
        self._threshold = threshold
        # Total number of tasks
        self._total_tasks = total_tasks
        # List of CLoRALayer module names
        self._module_names: Sequence[str] = []
        # Rank of each CLoRALayer module
        self._ranks: Dict[str, int] = {}
        # Gradient projection memory
        self._gpm: Optional[Dict[str, Tensor]] = None

    def before_training(self, strategy: Any, *args, **kwargs) -> Any:
        if not isinstance(strategy, BaseSGDTemplate):
            raise TypeError("PluginInfLoRA requires BaseSGDTemplate strategies.")
        if len(self._module_names) == 0:
            logger.info("Initializing PluginInfLoRA. Freezing LoRA A parameters.")
            self._module_names = []
            for name, module in iter_named_adapters(strategy.model):
                if not isinstance(module, CLoRALayer):
                    raise TypeError(
                        f"PluginInfLoRA can only be used with CLoRALayer modules, "
                        f"but found {type(module)} in {name}."
                    )
                self._module_names.append(name)
                module.clora_A.requires_grad_(False)
                self._ranks[name] = module.rank

        if not self._module_names:
            raise ValueError(
                "No CLoRALayer modules found in the model. PluginInfLoRA requires at least one CLoRALayer."
            )

    def before_training_exp(self, strategy: BaseSGDTemplate, *args, **kwargs) -> Any:
        logger.info("Creating orthogonal projection matrix for LoRA A parameters.")
        module_inputs = self.sample_module_inputs(strategy)
        lora_A = orthogonal_projection_matrix(module_inputs, self._gpm, self._ranks)
        for name, data in lora_A.items():
            module = strategy.model.get_submodule(name)
            if not isinstance(module, CLoRALayer):
                raise TypeError(
                    f"Expected CLoRALayer in {name}, but found {type(module)}."
                )
            assert module.clora_A.shape == data.shape, (
                f"Shape mismatch for {name}: expected {module.clora_A.shape}, "
                f"but got {data.shape}."
            )
            module.clora_A.data = data.to(strategy.device)
            module.clora_A.requires_grad_(False)

    def after_training_exp(self, strategy: BaseSGDTemplate, *args, **kwargs) -> Any:
        task = strategy.clock.train_exp_counter

        module_inputs = self.sample_module_inputs(strategy)
        threshold = threshold_schedule(task, self._total_tasks, self._threshold)
        if self._gpm is None:
            logger.info(f"Creating GPM threshold={threshold}")
            self._gpm = create_gpm(module_inputs, threshold=threshold)
        else:
            logger.info(f"Updating GPM threshold={threshold}")
            self._gpm = update_gpm(self._gpm, module_inputs, threshold=threshold)

        # Merge the previous adapter with the active adapter
        CLoRA.update_anchors(strategy.model)

        logger.info("Gradient Constraint Summary:")
        max_len = max(len(name) for name in self._module_names)
        for name, gpm in self._gpm.items():
            logger.info(f"{name:<{max_len}}: {gpm.shape[1]}/{gpm.shape[0]}")

    def sample_module_inputs(self, strategy):
        logger.info("Sampling module inputs for InfLoRA. This may take a while.")
        return sample_module_inputs(
            strategy.model,
            self._module_names,
            strategy.dataloader,  # type: ignore
            device=strategy.device,
        )
