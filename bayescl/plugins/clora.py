from typing import Any

from avalanche.training.plugins import SupervisedPlugin
from claiutil.peft import CLoRA
from loguru import logger
from torch.utils.tensorboard import SummaryWriter


class CLoRAPlugin(SupervisedPlugin):
    def __init__(self, beta: float, writer: SummaryWriter | None = None):
        self.beta = beta
        self.writer = writer

    def after_training_exp(self, strategy: Any, *args, **kwargs) -> Any:
        logger.info("Update CLoRA anchors")
        CLoRA.update_anchors(strategy.model)

    def before_backward(self, strategy: Any, *args, **kwargs) -> Any:
        reg_loss = CLoRA.self_regularization_loss(strategy.model)
        step = strategy.clock.train_iterations
        if self.writer is not None:
            self.writer.add_scalar("CLoRA/reg_loss", reg_loss.item(), step)
            self.writer.add_scalar("CLoRA/ce_loss", strategy.loss.item(), step)
        strategy.loss += self.beta * reg_loss
