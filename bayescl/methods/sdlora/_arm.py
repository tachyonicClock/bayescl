from dataclasses import dataclass

from bayescl.experiment import Experiment
from bayescl.methods._registry import ArmBase, register
from bayescl.methods.sdlora import SDLoRAConfig
from bayescl.spec import ExperimentSpec, NaiveConfig


@register("sdlora")
@dataclass
class SDLoRA(ArmBase):
    lr: float = 1e-3
    rank_per_task: int = 1

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
            peft=SDLoRAConfig(rank_per_task=self.rank_per_task),
            strategy=NaiveConfig(),
        )
        return Experiment(spec)
