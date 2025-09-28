from typing import Any

import torch
from avalanche.training.plugins import SupervisedPlugin


class TrainTaskMask(SupervisedPlugin):
    def __init__(self, mask: torch.Tensor, optimizer_fn):
        super().__init__()
        self.mask = mask
        self.optimizer_fn = optimizer_fn

    def before_training_exp(self, strategy: Any, *args, **kwargs) -> Any:
        strategy.optimizer = self.optimizer_fn()

    def before_training(self, strategy: Any, *args, **kwargs) -> Any:
        self.mask = self.mask.to(strategy.device)

    def after_forward(self, strategy: Any, *args, **kwargs) -> Any:
        t = strategy.clock.train_exp_counter
        strategy.mb_output = self.mask[t] * strategy.mb_output
