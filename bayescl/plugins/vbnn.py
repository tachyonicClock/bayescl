from typing import Any

import matplotlib.pyplot as plt
import torch
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import BaseSGDTemplate
from claiutil.vbnn import (
    get_model_kl_loss,
    get_posterior_state,
    iterate_variational_parameters,
    set_prior_state,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from seaborn import histplot
from torch.utils.tensorboard import SummaryWriter
from loguru import logger


class VBNNPlugin(SupervisedPlugin):
    def __init__(
        self, beta: float, bayes_eval_samples: int, writer: SummaryWriter | None = None
    ):
        super().__init__()
        self.beta = beta
        self.writer = writer
        self.bayes_eval_samples = bayes_eval_samples

    def before_backward(self, strategy: BaseSGDTemplate, *args, **kwargs) -> Any:
        # Scale the KL divergence by the number of samples in the dataset so that
        # it is in the same scale as the cross-entropy loss.
        # The raw kl divergence is independent of the data batch size.
        kl = get_model_kl_loss(strategy.model) / len(strategy.experience.dataset)
        if self.writer is not None:
            i = strategy.clock.train_iterations
            self.writer.add_scalar("VBNN/kl_loss", kl.item(), i)
            self.writer.add_scalar("VBNN/beta_kl_loss", self.beta * kl.item(), i)
            self.writer.add_scalar("VBNN/ce_loss", strategy.loss.item(), i)
        strategy.loss += self.beta * kl

    def after_training_epoch(self, strategy: BaseSGDTemplate, *args, **kwargs) -> Any:
        assert self.writer is not None
        fig = self.visualize_mu_sigma(strategy, strategy.model.model.vit) # type: ignore
        self.writer.add_figure("VBNN/posterior_vit", fig, strategy.clock.train_iterations)
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
        set_prior_state(strategy.model, get_posterior_state(strategy.model))

    def after_eval_forward(self, strategy: Any, *args, **kwargs) -> Any:
        x, y, _ = strategy.mbatch
        for _ in range(self.bayes_eval_samples):
            strategy.mb_output += strategy.model(x)
        strategy.mb_output /= self.bayes_eval_samples + 1
