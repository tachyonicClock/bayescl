from dataclasses import dataclass

from bayescl.experiment import Experiment
from bayescl.methods._registry import ArmBase, register
from bayescl.methods.inflora import InfLoRAConfig
from bayescl.spec import ExperimentSpec, NaiveConfig


@register("inflora")
@dataclass
class InfLoRA(ArmBase):
    lr: float = 1e-3
    rank: int = 10
    threshold_start: float = 0.90
    threshold_end: float = 0.98
    max_activation_batches: int | None = 16
    activation_batch_size: int = 32

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
            peft=InfLoRAConfig(
                rank=self.rank,
                threshold_start=self.threshold_start,
                threshold_end=self.threshold_end,
                max_activation_batches=self.max_activation_batches,
                activation_batch_size=self.activation_batch_size,
            ),
            strategy=NaiveConfig(),
        )
        return Experiment(spec)
