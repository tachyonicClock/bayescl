from dataclasses import dataclass, replace

from bayescl.experiment import Experiment
from bayescl.methods._registry import ArmBase, register
from bayescl.methods.lora import LoRAConfig
from bayescl.spec import ExperimentSpec, NaiveConfig, RWalkConfig


@register("rwalk")
@dataclass
class RWalk(ArmBase):
    lr: float = 1e-3
    r: int = 10
    ewc_lambda: float = 0.1
    ewc_alpha: float = 0.9
    delta_t: int = 10

    @staticmethod
    def suggest_config(trial, base):
        return replace(
            base,
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            ewc_lambda=trial.suggest_float("ewc_lambda", 0.0, 1.0),
            ewc_alpha=trial.suggest_float("ewc_alpha", 0.0, 1.0),
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
            peft=LoRAConfig(r=self.r),
            strategy=NaiveConfig(),
            rwalk=RWalkConfig(
                ewc_lambda=self.ewc_lambda,
                ewc_alpha=self.ewc_alpha,
                delta_t=self.delta_t,
            ),
        )
        return Experiment(spec)
