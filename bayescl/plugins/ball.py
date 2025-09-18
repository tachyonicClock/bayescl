from typing import Any

import matplotlib.pyplot as plt
import torch
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import BaseSGDTemplate
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from seaborn import histplot
from torch import BoolTensor, nn
from torch.utils.tensorboard import SummaryWriter

from bayescl.vbnn import (
    VariationalLinear,
    get_model_kl_loss,
    get_posterior_state,
    iterate_variational_parameters,
    set_prior_state,
)


class BALLPlugin(SupervisedPlugin):
    def __init__(
        self,
        beta: float,
        bayes_eval_samples: int,
        writer: SummaryWriter | None = None,
        first_task_beta: float | None = None,
    ):
        super().__init__()
        self.beta = beta
        self.first_task_beta = first_task_beta if first_task_beta is not None else beta
        self.writer = writer
        self.bayes_eval_samples = bayes_eval_samples
        assert self.bayes_eval_samples >= 1

    def before_backward(self, strategy: BaseSGDTemplate, *args, **kwargs) -> Any:
        train_task_id = strategy.clock.train_exp_counter
        backbone: nn.Module = strategy.model.model.vit  # type: ignore
        classifier: nn.Module = strategy.model.model.classifier  # type: ignore

        kl = get_model_kl_loss(backbone)  # type: ignore
        if isinstance(classifier, VariationalLinear):
            # mask the kl divergence so only the current task's output heads are regularized
            mask: BoolTensor = strategy.mask[train_task_id]  # type: ignore
            kl += classifier.weight.kl_divergences()[mask].sum()
            if classifier.bias is not None:
                kl += classifier.bias.kl_divergences()[mask].sum()

        # Scale the KL divergence by the number of samples in the dataset so that
        # it is in the same scale as the cross-entropy loss.
        # The raw kl divergence is independent of the data batch size.
        kl /= len(strategy.adapted_dataset)

        # sometimes we want to use a smaller beta in the first task to avoid
        # under regularization in subsequent tasks
        beta = self.beta if train_task_id != 0 else self.first_task_beta

        if self.writer is not None:
            i = strategy.clock.train_iterations
            self.writer.add_scalar("VBNN/kl_loss", kl.item(), i)
            self.writer.add_scalar("VBNN/beta_kl_loss", beta * kl.item(), i)
            self.writer.add_scalar("VBNN/ce_loss", strategy.loss.item(), i)
        strategy.loss += beta * kl

    def after_training_epoch(self, strategy: BaseSGDTemplate, *args, **kwargs) -> Any:
        assert self.writer is not None
        fig = self.visualize_mu_sigma(strategy, strategy.model.model.vit)  # type: ignore
        self.writer.add_figure(
            "VBNN/posterior_vit", fig, strategy.clock.train_iterations
        )
        plt.close(fig)
        # fig = self.visualize_mu_sigma(strategy, strategy.model.model.classifier) # type: ignore
        # self.writer.add_figure("VBNN/posterior_classifier", fig, strategy.clock.train_iterations)
        # plt.close(fig)

    def visualize_mu_sigma(self, strategy, model):
        ax_mu: Axes
        ax_sigma: Axes
        fig: Figure
        fig, (ax_mu, ax_sigma) = plt.subplots(1, 2, figsize=(5, 2.5), tight_layout=True)

        mu_list = []
        sigma_list = []
        for name, param in iterate_variational_parameters(model):
            mu_list.append(param.mu.detach().cpu().flatten())
            sigma_list.append(param.sigma().detach().cpu().flatten())

        histplot(torch.concat(mu_list).numpy(), ax=ax_mu, stat="density")
        ax_mu.set_ylabel(
            f"epoch={strategy.clock.train_exp_epochs} t={strategy.clock.train_exp_counter}"
        )
        ax_mu.set_xlabel(r"$\mu$")
        ax_mu.set_xlim(-0.5, 0.5)
        ax_mu.set_ylim(0, 15)

        histplot(torch.concat(sigma_list).numpy(), ax=ax_sigma, stat="density")
        ax_sigma.set_xlabel(r"$\sigma$")
        ax_sigma.set_xlim(0.0, 1.5)
        ax_sigma.set_ylim(0, 15)
        return fig

    def after_training_exp(self, strategy: Any, *args, **kwargs) -> Any:
        logger.info("Setting prior to posterior after training experience.")
        backbone: nn.Module = strategy.model.model.vit  # type: ignore
        set_prior_state(backbone, get_posterior_state(backbone))

    def after_eval_forward(self, strategy: Any, *args, **kwargs) -> Any:
        x, _, _ = strategy.mbatch
        y_logit = strategy.mb_output
        for _ in range(self.bayes_eval_samples - 1):
            y_logit += strategy.model(x)
        strategy.mb_output = y_logit / self.bayes_eval_samples
