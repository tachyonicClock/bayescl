from dataclasses import dataclass

from bayescl.experiment import Experiment
from bayescl.methods._registry import ArmBase, register
from bayescl.methods.lora import LoRAConfig
from bayescl.spec import ExperimentSpec, NaiveConfig


@register("lora")
@dataclass
class LoRA(ArmBase):
    lr: float = 1e-3
    r: int = 10
    lora_alpha: int = 1
    lora_dropout: float = 0.0

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
            peft=LoRAConfig(
                r=self.r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout
            ),
            strategy=NaiveConfig(),
        )
        return Experiment(spec)
