"""Strategy, PEFT, and plugin configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from bayescl.methods.ball import BALLConfig
from bayescl.methods.clora import CLoRAConfig
from bayescl.methods.inflora import InfLoRAConfig
from bayescl.methods.lora import LoRAConfig
from bayescl.methods.sdlora import SDLoRAConfig
from bayescl.methods.tball import TBALLConfig

# --- Strategy configs (only the two actually used survive the migration) ---


@dataclass
class NaiveConfig:
    type: ClassVar[str] = "Naive"


@dataclass
class VCLConfig:
    """Variational Continual Learning strategy.

    Requires a model or PEFT module that implements Bayesian layers.
    """

    beta: float = 1.0
    """Hyperparameter weighting the KL divergence loss."""
    train_samples: int = 1
    """Number of samples for each step of training."""
    test_samples: int = 5
    """Number of samples for each step of testing."""
    softmax_avg: bool = False
    """If true, softmax then average, otherwise average then softmax."""
    train_mask: bool = True
    """Should a mask be used during training."""

    type: ClassVar[str] = "VCL"

    def __post_init__(self) -> None:
        if self.train_samples < 1:
            raise ValueError("train_samples must be at least 1")
        if self.test_samples < 1:
            raise ValueError("test_samples must be at least 1")


StrategyConfig = NaiveConfig | VCLConfig

PeftConfig = (
    BALLConfig | TBALLConfig | LoRAConfig | CLoRAConfig | InfLoRAConfig | SDLoRAConfig
)


# --- Regularization plugin configs ---


@dataclass
class EWCConfig:
    ewc_lambda: float
    decay_factor: float


@dataclass
class RWalkConfig:
    ewc_lambda: float
    """Weight of the penalty in the total loss."""
    ewc_alpha: float
    """Moving-average factor for the importance matrix. Must be in [0, 1]."""
    delta_t: int
    """Iteration interval at which parameter scores are updated."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.ewc_alpha <= 1.0:
            raise ValueError("ewc_alpha must be in [0, 1]")



