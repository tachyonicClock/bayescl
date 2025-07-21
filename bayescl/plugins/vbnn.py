from typing import Any

from avalanche.training.plugins import SupervisedPlugin
from claiutil.vbnn import get_model_kl_loss, get_posterior_state, set_prior_state
from torch.utils.tensorboard import SummaryWriter


class VBNNPlugin(SupervisedPlugin):
    def __init__(self, beta: float, writer: SummaryWriter | None = None):
        super().__init__()
        self.beta = beta
        self.writer = writer

    def before_backward(self, strategy: Any, *args, **kwargs) -> Any:
        kl = get_model_kl_loss(strategy.model)
        if self.writer is not None:
            self.writer.add_scalar(
                "VBNN/kl_loss", kl.item(), strategy.clock.train_iterations
            )
            self.writer.add_scalar(
                "VBNN/ce_loss", strategy.loss.item(), strategy.clock.train_iterations
            )
        strategy.loss += self.beta * kl

    def after_training_exp(self, strategy: Any, *args, **kwargs) -> Any:
        set_prior_state(strategy.model, get_posterior_state(strategy.model))
