import numpy as np
import torch
from bnn.nn.modules import FFGLinear
from capymoa.base import BatchClassifier
from capymoa.ocl.base import TrainTaskAware
from capymoa.ocl.datasets import SplitCIFAR100ViT
from capymoa.ocl.evaluation import ocl_train_eval_loop
from capymoa.ocl.util.data import class_schedule_to_task_mask
from capymoa.stream import Schema
from loguru import logger
from matplotlib import pyplot as plt
from torch.optim import Adam
from vbll.layers.classification import DiscClassification, gaussian_kl, Normal, DenseNormal, LowRankNormal
from torch.distributions import kl_divergence, Distribution, register_kl, LowRankMultivariateNormal
from copy import deepcopy
import optuna
import os


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


@register_kl(LowRankMultivariateNormal, Normal)
def _kl_lowranknormal_normal(p: LowRankMultivariateNormal, q: Normal):
    return gaussian_kl(p, (q.scale**2).cpu().item())


def loss_fn(model: DiscClassification, prior: Distribution, x, y):
    noise = model.noise()

    kl_term = kl_divergence(model.W(), prior).sum()

    # kl_term = gaussian_kl(model.W(), model.prior_scale)
    # assert torch.allclose(kl_term, kl_term_actual), f"{kl_term} != {kl_term_actual}"

    wishart_term = (model.dof * noise.logdet_precision - 0.5 * model.wishart_scale * noise.trace_precision)

    total_elbo = torch.mean(model.softmax_bound(x, y))
    total_elbo += model.regularization_weight * (wishart_term - kl_term)
    return -total_elbo

@torch.no_grad()
def new_distribution(model) -> Distribution:
    cov_diag = torch.exp(model.W_logdiag.detach().clone())
    W_mean = model.W_mean.detach().clone()

    if model.W_dist == Normal:
        cov = model.W_dist(W_mean, cov_diag)
    elif model.W_dist == DenseNormal:
        W_offdiag = model.W_offdiag.detach().clone()
        tril = torch.tril(W_offdiag, diagonal=-1) + torch.diag_embed(cov_diag)
        cov = model.W_dist(W_mean, tril)
    elif model.W_dist == LowRankNormal:
        W_offdiag = model.W_offdiag.detach().clone()
        cov = model.W_dist(W_mean, W_offdiag, cov_diag)
    else:
        raise ValueError(f"Unsupported distribution: {model.W_dist}")
    return cov


class VCLHead(BatchClassifier, TrainTaskAware):
    def __init__(
        self,
        schema: Schema,
        lr: float,
        reg_weight: float = 1.0,
        cov_rank=10,
        
    ):
        super().__init__(schema)
        self.device = torch.device("cuda")
        self.model = DiscClassification(
            in_features=schema.get_num_attributes(),
            out_features=schema.get_num_classes(),
            regularization_weight=reg_weight,
            # parameterization="diagonal",
            parameterization="lowrank" if cov_rank > 0 else "diagonal",
            cov_rank=cov_rank
        ).to(self.device)
        self.prior = Normal(0, torch.tensor(self.model.prior_scale**0.5).to(self.device))
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.log_loss = []

    def batch_train(self, x: torch.Tensor, y: torch.Tensor) -> None:
        # W = LowRankNormal(self.model.W_mean, self.model.W_offdiag, torch.exp(self.model.W_logdiag))

        # assert self.model.W_offdiag.device == x.device
        # assert self.model.W_mean.device == x.device
        # assert self.model.W_logdiag.device == x.device
        # assert W.cov_factor == x.device
        # out = self.model(x)
        # loss = out.train_loss_fn(y)
        loss = loss_fn(self.model, self.prior, x, y)
        loss.backward()

        self.log_loss.append(loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad()



    def on_train_task(self, task_id: int) -> None:
        if task_id > 0:
            logger.info(f"Updating prior for task {task_id}.")
            distribution: Distribution = new_distribution(self.model)
            self.prior = distribution

    def batch_predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).predictive.probs


def objective(trial: optuna.Trial) -> float:
    torch.manual_seed(0)
    np.random.seed(0)

    stream = SplitCIFAR100ViT()
    # task_mask = class_schedule_to_task_mask(stream.task_schedule, stream.num_classes)
    learner = VCLHead(
        stream.schema,
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        reg_weight=trial.suggest_float("beta", 0.1, 10.0, log=True)/len(stream.train_tasks[0]),
        cov_rank=trial.suggest_categorical("cov_rank", [0, 5, 10, 20]),
    )
    # schedule = flatten(stream.task_schedule)
    results = ocl_train_eval_loop(
        learner,
        stream.train_loaders(128, True),
        stream.test_loaders(128),
        progress_bar=True,
        epochs=trial.suggest_categorical("epochs", [1, 5, 10])
    )

    print(f"accuracy_seen_avg {results.accuracy_seen_avg * 100:.2f}")
    print(f"accuracy_final    {results.accuracy_final * 100:.2f}")
    print("accuracy_seen    ", [f"{v * 100:.2f}" for v in results.accuracy_seen])

    return results.accuracy_seen_avg


study = optuna.create_study(
    direction="maximize",
    study_name="vbll",
    storage=os.getenv("OPTUNA_STORAGE"),
    load_if_exists=True,
)
study.optimize(objective, n_trials=100)
    

# fig, ax = plt.subplots(1, 1, figsize=(4, 4))
# ax.imshow(results.accuracy_matrix.T, origin="lower")
# ax.set_title("Accuracy Matrix")
# ax.set_xlabel("Task")
# ax.set_ylabel("Evaluated Task")
# fig.savefig("vcl_accuracy_matrix.png")

# fig, ax = plt.subplots(1, 1, figsize=(4, 4))
# cm: np.ndarray = results.class_cm[-1][schedule][:, schedule]
# print("above diagonal:", np.triu(cm, k=1).sum() / cm.sum())
# print("below diagonal:", np.tril(cm, k=-1).sum() / cm.sum())
# ax.imshow(cm)
# ax.set_title("Class Confusion Matrix")
# ax.set_xlabel("Predicted Label")
# ax.set_ylabel("True Label")
# fig.savefig("cm.png")

# fig, ax = plt.subplots(1, 1, figsize=(6, 4))
# ax.plot(learner.log_loss)
# # ax.set_yscale("log")
# ax.set_title("Training Loss")
# ax.set_xlabel("Iteration")
# ax.set_ylabel("Loss")
# fig.savefig("vcl_training_loss.png")
