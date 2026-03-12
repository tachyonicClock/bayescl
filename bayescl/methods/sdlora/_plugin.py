from typing import Any

from avalanche.training.plugins import SupervisedPlugin
from loguru import logger

from ._module import set_task


class SDLoRAPlugin(SupervisedPlugin):
    def before_training_exp(self, strategy: Any, *args, **kwargs) -> Any:
        logger.info(f"Setting SD-LoRA task to {strategy.clock.train_exp_counter}")
        set_task(strategy.model, strategy.clock.train_exp_counter)
