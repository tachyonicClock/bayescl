from typing import Any

from avalanche.training.plugins import SupervisedPlugin
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from ._config import CLoRAConfig
from ._module import clora_loss, update_anchors


class CLoRAPlugin(SupervisedPlugin):
    def __init__(self, config: CLoRAConfig, writer: SummaryWriter):
        self.config = config
        self.writer = writer

    def after_training_exp(self, strategy: Any, *args, **kwargs) -> Any:
        logger.info("Updating CLoRA Anchors")
        task_index = strategy.clock.train_exp_counter
        update_anchors(strategy.model, task_index)

    def before_backward(self, strategy: Any, *args, **kwargs) -> Any:
        loss = clora_loss(strategy.model)
        strategy.loss += self.config.lambda_ * loss
        self.writer.add_scalar(
            "train/clora_loss", loss.item(), strategy.clock.train_iterations
        )
