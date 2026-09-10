from dataclasses import dataclass, replace

from bayescl.experiment import Experiment
from bayescl.methods._registry import ArmBase, register
from bayescl.methods.lora import LoRAConfig
from bayescl.spec import EWCConfig, ExperimentSpec, NaiveConfig


@register("ewc")
@dataclass
class EWC(ArmBase):
    lr: float = 1e-3
    r: int = 10
    ewc_lambda: float = 1.0
    decay_factor: float = 0.9

    @staticmethod
    def suggest_config(trial, base):
        return replace(
            base,
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            ewc_lambda=trial.suggest_float("ewc_lambda", 0.0, 10.0),
            decay_factor=trial.suggest_float("decay_factor", 0.8, 1.0),
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
            ewc=EWCConfig(ewc_lambda=self.ewc_lambda, decay_factor=self.decay_factor),
        )
        return Experiment(spec)
