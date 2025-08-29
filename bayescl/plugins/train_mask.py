from typing import Any

import torch
from avalanche.training.plugins import SupervisedPlugin
from loguru import logger


class TrainTaskMask(SupervisedPlugin):
    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.mask = mask

    def before_training(self, strategy: Any, *args, **kwargs) -> Any:
        self.mask = self.mask.to(strategy.device)

    def before_training_exp(self, strategy: Any, *args, **kwargs) -> Any:
        # Momentum optimizer can cause issues in continual learning if the optimizer
        # state is not reset. This is because the optimizer state may contain
        # information from previous tasks that can interfere with the current task. We
        # reset the optimizer state at the beginning of each training experience.
        optimizer = strategy.optimizer
        assert isinstance(optimizer, torch.optim.Optimizer)
        t = strategy.clock.train_exp_counter
        if t == 0:
            logger.info("Saving optimizer state")
            self.optimizer_state = optimizer.state_dict()
        else:
            logger.info("Restoring optimizer state")
            optimizer.load_state_dict(self.optimizer_state)

    def after_forward(self, strategy: Any, *args, **kwargs) -> Any:
        t = strategy.clock.train_exp_counter
        strategy.mb_output = self.mask[t] * strategy.mb_output
