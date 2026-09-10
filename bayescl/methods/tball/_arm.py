from dataclasses import dataclass, replace

from bayescl.experiment import Experiment
from bayescl.methods._registry import ArmBase, register
from bayescl.methods.tball import TBALLConfig
from bayescl.spec import ExperimentSpec, VCLConfig


@register("tball")
@dataclass
class TBALL(ArmBase):
    lr: float = 1e-3
    rank: int = 10
    alpha: float = 1.0
    prior_mean: float = 0.0
    prior_weight_sd: float = 1.0
    init_sd: float = 0.5
    nonlinearity_scale: float = 1.0
    bnn: str = "FFG"
    bias: bool = False
    # VCL strategy
    beta: float = 1.0
    train_samples: int = 1
    test_samples: int = 5

    @staticmethod
    def suggest_config(trial, base):
        return replace(
            base,
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            beta=trial.suggest_float("beta", 0.0, 2.0),
        )

    def _peft(self) -> TBALLConfig:
        return TBALLConfig(
            rank=self.rank,
            alpha=self.alpha,
            prior_mean=self.prior_mean,
            prior_weight_sd=self.prior_weight_sd,
            init_sd=self.init_sd,
            nonlinearity_scale=self.nonlinearity_scale,
            bnn=self.bnn,
            bias=self.bias,
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
                use_local_ce=False,  # TBALL implements local CE internally
            ),
            peft=self._peft(),
            strategy=VCLConfig(
                beta=self.beta,
                train_samples=self.train_samples,
                test_samples=self.test_samples,
            ),
        )
        return Experiment(spec)


@register("tball-mnd")
@dataclass
class TBALLMND(TBALL):
    # Registry variant: identical to TBALL except the Bayesian core is a matrix
    # normal distribution. ``suggest_config`` / ``build`` are inherited unchanged.
    bnn: str = "MND"
