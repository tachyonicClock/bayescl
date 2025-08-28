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

# from torch.distributions.kl import kl_divergence


class MyModel(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.hidden = FFGLinear(input_size, 768)
        self.head = torch.nn.Linear(768, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.hidden(x)) + x
        x = self.head(x)
        return x


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
        self.model = MyModel(
            schema.get_num_attributes(),
            schema.get_num_classes(),
        ).to(self.device)
        # self.model.hidden.requires_grad_(False)

        self.task_mask = task_mask.to(self.device)
        self.train_task_id = 0
        self.lr = lr
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.mc_samples = mc_samples
        self.beta = beta
        self.log_nll = []
        self.log_kl = []
        self.log_beta_kl = []
        self.log_loss = []

    def kl_divergence(self) -> torch.Tensor:
        kl = torch.tensor(0.0, device=self.device)
        for module in self.model.modules():
            if isinstance(module, FFGLinear):
                kl += module.kl_divergence()
        return kl

    def batch_train(self, x: torch.Tensor, y: torch.Tensor) -> None:
        mask = self.task_mask[self.train_task_id]
        nll = torch.zeros(1, device=self.device)
        for _ in range(self.mc_samples):
            y_hat = mask * self.model(x)
            nll += torch.nn.functional.cross_entropy(y_hat, y)
        nll /= self.mc_samples

        # kl = (
        #     self.head.weight.kl_divergences()[mask].sum()
        #     + self.head.bias.kl_divergences()[mask].sum()
        # )
        # kl = (
        #     kl_divergence(self.head.weight_dist, self.head.prior_weight_dist)[
        #         mask
        #     ].sum()
        #     + kl_divergence(self.head.bias_dist, self.head.prior_bias_dist)[mask].sum()
        #     if self.head.bias_dist and self.head.prior_bias_dist
        #     else 0
        # ) / len(x)
        kl = self.kl_divergence() / len(x)

        beta_kl = self.beta * kl
        loss = nll + beta_kl
        self.log_nll.append(nll.item())
        self.log_kl.append(kl.item())
        self.log_loss.append(loss.item())
        self.log_beta_kl.append(beta_kl.item())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def on_train_task(self, task_id: int):
        self.train_task_id = task_id

        if self.train_task_id > 0:
            logger.info(f"Updating prior for task {task_id}.")
            # mask = self.task_mask[task_id - 1]
            # self.head.weight.prior_mu[mask] = self.head.weight.mu[mask].detach().clone()
            # self.head.weight.prior_sigma[mask] = self.head.weight.sigma()[mask].detach().clone()

            # Only priors associated with previous task's classes
            # self.head.prior_weight_mean[mask] = self.head.weight_mean[mask].detach().clone()
            # self.head.prior_weight_sd[mask] = self.head.weight_sd[mask].detach().clone()

            # Update all priors
            for module in self.model.modules():
                if isinstance(module, FFGLinear):
                    logger.info(f"Updating prior for module {module}.")
                    module.prior_weight_mean = module.weight_mean.detach().clone()
                    module.prior_weight_sd = module.weight_sd.detach().clone()
                    print(
                        module.weight_sd.min(),
                        module.weight_sd.max(),
                        module.weight_sd.mean(),
                        module.weight_sd.std(),
                    )
                    if module.has_bias:
                        module.prior_bias_mean = module.bias_mean.detach().clone()
                        module.prior_bias_sd = module.bias_sd.detach().clone()
                    assert module.kl_divergence() == 0

            self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def batch_predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(self.model(x), dim=1)


torch.manual_seed(0)
np.random.seed(0)

stream = SplitCIFAR100ViT()
task_mask = class_schedule_to_task_mask(stream.task_schedule, stream.num_classes)
learner = VCLHead(
    stream.schema,
    lr=1e-2,
    mc_samples=5,
    task_mask=task_mask,
    # beta=1.0/ len(stream.train_tasks[0]),
    beta=0.01 / len(stream.train_tasks[0]),
    # beta=1.0
    # beta=
)
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
nll_ax.plot(learner.log_beta_kl, label="beta * KL")
nll_ax.legend()
nll_ax.set_ylabel("NLL")
kl_ax.plot(learner.log_kl)
kl_ax.set_ylabel("KL")
loss_ax.plot(learner.log_loss)
loss_ax.set_ylabel("Loss")

fig.savefig("vcl_loss.png")
