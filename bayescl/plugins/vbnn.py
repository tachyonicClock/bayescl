from typing import Any

from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import BaseSGDTemplate
from claiutil.vbnn import get_model_kl_loss, get_posterior_state, set_prior_state
from torch.utils.tensorboard import SummaryWriter


class VBNNPlugin(SupervisedPlugin):
    def __init__(
        self, beta: float, bayes_eval_samples: int, writer: SummaryWriter | None = None
    ):
        super().__init__()
        self.beta = beta
        self.writer = writer
        self.bayes_eval_samples = bayes_eval_samples

    def before_backward(self, strategy: BaseSGDTemplate, *args, **kwargs) -> Any:
        kl = get_model_kl_loss(strategy.model, len(strategy.experience.dataset))
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

    def after_eval_forward(self, strategy: Any, *args, **kwargs) -> Any:
        x, y, _ = strategy.mbatch
        for _ in range(self.bayes_eval_samples):
            strategy.mb_output += strategy.model(x)
        strategy.mb_output /= self.bayes_eval_samples + 1
