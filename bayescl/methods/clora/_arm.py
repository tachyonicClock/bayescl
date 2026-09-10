from dataclasses import dataclass, replace

from bayescl.experiment import Experiment
from bayescl.methods._registry import ArmBase, register
from bayescl.methods.clora import CLoRAConfig
from bayescl.spec import ExperimentSpec, NaiveConfig


@register("clora")
@dataclass
class CLoRA(ArmBase):
    lr: float = 1e-3
    rank: int = 10
    alpha: float = 1.0
    lambda_: float = 1.0

    @staticmethod
    def suggest_config(trial, base):
        return replace(
            base,
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            lambda_=trial.suggest_float("lambda_", 0.01, 100.0, log=True),
            alpha=trial.suggest_float("alpha", 0.5, 2.0),
        )

    def build(self, *, dataset, scale, seed, validation, run_dir, dataset_root):
        spec = ExperimentSpec(
            **self.base_spec(
                dataset=dataset,
                scale=scale,
                seed=seed,
                validation=validation,
                run_dir=run_dir,
                dataset_root=dataset_root,
            ),
            peft=CLoRAConfig(rank=self.rank, alpha=self.alpha, lambda_=self.lambda_),
            strategy=NaiveConfig(),
        )
        return Experiment(spec)
