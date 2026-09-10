from dataclasses import dataclass, replace

from bayescl.experiment import Experiment
from bayescl.methods._registry import ArmBase, register
from bayescl.methods.ball import BALLConfig
from bayescl.spec import ExperimentSpec, VCLConfig
from bayescl.vbnn import VBNNConfig


@register("ball")
@dataclass
class BALL(ArmBase):
    lr: float = 1e-3
    r: int = 10
    lora_alpha: int = 1
    dropout: float = 0.0
    bll: bool = False
    # VCL strategy
    beta: float = 1.0
    train_samples: int = 1
    test_samples: int = 5
    # variational prior / init
    prior_mean: float = 0.0
    prior_sd: float = 1.0
    init_sd: float = 1e-3
    init_sd_sd: float = 1e-5

    @staticmethod
    def suggest_config(trial, base):
        return replace(
            base,
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            beta=trial.suggest_float("beta", 0.0, 2.0),
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
                use_local_ce=False,  # BALL implements local CE internally
            ),
            peft=BALLConfig(
                r=self.r,
                lora_alpha=self.lora_alpha,
                dropout=self.dropout,
                bll=self.bll,
                vbnn=VBNNConfig(
                    prior_mean=self.prior_mean,
                    prior_sd=self.prior_sd,
                    init_sd=self.init_sd,
                    init_sd_sd=self.init_sd_sd,
                ),
            ),
            strategy=VCLConfig(
                beta=self.beta,
                train_samples=self.train_samples,
                test_samples=self.test_samples,
            ),
        )
        return Experiment(spec)
