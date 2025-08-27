from capymoa.ocl.datasets import SplitCIFAR100ViT
from capymoa.ocl.util.data import class_schedule_to_task_mask
from capymoa.ocl.base import TrainTaskAware
from capymoa.ocl.evaluation import ocl_train_eval_loop
from capymoa.stream import Schema
from capymoa.base import BatchClassifier
from claiutil.vbnn import (
    VariationalLinear,
    get_model_kl_loss,
    get_posterior_state,
    set_prior_state,
)
from torch.optim import Adam
import torch
from loguru import logger
from matplotlib import pyplot as plt
import numpy as np


class VCLHead(BatchClassifier, TrainTaskAware):
    def __init__(
        self,
        schema: Schema,
        mc_samples: int,
        lr: float,
        beta: float,
        task_mask: torch.Tensor,
    ):
        super().__init__(schema)
        self.device = torch.device("cuda")
        self.head = VariationalLinear(
            dim_in=schema.get_num_attributes(),
            dim_out=schema.get_num_classes(),
        ).to(self.device)
        self.task_mask = task_mask.to(self.device)
        self.train_task_id = 0
        self.lr = lr
        self.optimizer = Adam(self.head.parameters(), lr=lr)
        self.mc_samples = mc_samples
        self.beta = beta
        self.log_nll = []
        self.log_kl = []
        self.log_loss = []

    def batch_train(self, x: torch.Tensor, y: torch.Tensor) -> None:
        mask = self.task_mask[self.train_task_id]
        nll = torch.zeros(1, device=self.device)
        for _ in range(self.mc_samples):
            y_hat = mask * self.head(x)
            nll += torch.nn.functional.cross_entropy(y_hat, y)
        nll /= self.mc_samples


        kl = (self.head.weight.kl_divergences()[mask].sum() + self.head.bias.kl_divergences()[mask].sum())
        loss = nll + self.beta * kl
        self.log_nll.append(nll.item())
        self.log_kl.append(kl.item())
        self.log_loss.append(loss.item())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def on_train_task(self, task_id: int):
        self.train_task_id = task_id

        if self.train_task_id > 0:
            logger.info(f"Updating prior for task {task_id}.")
            mask = self.task_mask[task_id - 1]
            self.head.weight.prior_mu[mask] = self.head.weight.mu[mask].detach().clone()
            self.head.weight.prior_sigma[mask] = self.head.weight.sigma()[mask].detach().clone()
            self.optimizer = Adam(self.head.parameters(), lr=self.lr)

    def batch_predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(self.head(x), dim=1)


torch.manual_seed(0)
np.random.seed(0)

stream = SplitCIFAR100ViT()
task_mask = class_schedule_to_task_mask(stream.task_schedule, stream.num_classes)
learner = VCLHead(stream.schema, lr=1e-2, mc_samples=5, task_mask=task_mask, beta=1.0/len(stream.train_tasks[0]))
results = ocl_train_eval_loop(
    learner,
    stream.train_loaders(128),
    stream.test_loaders(128),
    progress_bar=True,
)

print(f"accuracy_seen_avg {results.accuracy_seen_avg * 100:.2f}")
print(f"accuracy_final    {results.accuracy_final * 100:.2f}")
print("accuracy_seen    ", [f"{v * 100:.2f}" for v in results.accuracy_seen])

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.imshow(results.accuracy_matrix.T, origin="lower")
ax.set_title("Accuracy Matrix")
ax.set_xlabel("Task")
ax.set_ylabel("Evaluated Task")
fig.savefig("vcl_accuracy_matrix.png")


fig, (nll_ax, kl_ax, loss_ax) = plt.subplots(
    3, 1, figsize=(4, 8), sharex=True, tight_layout=True
)

nll_ax.plot(learner.log_nll)
nll_ax.set_ylabel("NLL")
kl_ax.plot(learner.log_kl)
kl_ax.set_ylabel("KL")
loss_ax.plot(learner.log_loss)
loss_ax.set_ylabel("Loss")

fig.savefig("vcl_loss.png")
