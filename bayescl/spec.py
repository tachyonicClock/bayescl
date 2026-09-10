"""Flat experiment specification.

Replaces the old god-object ``bayescl.config.Config``. An :class:`ExperimentSpec`
is assembled by an arm's ``build`` method (see :mod:`bayescl.methods`) and consumed
by :class:`bayescl.experiment.Experiment`, :func:`bayescl.benchmark.get_benchmark`
and :func:`bayescl.model.get_model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


@dataclass
class ExperimentSpec:
    # --- benchmark ---
    dataset: str
    """Avalanche scenario name, e.g. ``"CIFAR100"``."""
    n_tasks: int
    shuffle: bool
    dataset_root: Path
    standardize: bool
    validation: bool
    """Replace the test set with a validation split (used during ``tune``)."""

    # --- backbone ---
    backbone_name: str
    freeze_backbone: bool
    adapter_filter: str
    head_module: str

    # --- optimisation ---
    epochs: int
    train_mb_size: int
    eval_mb_size: int | None
    num_workers: int

    # --- output ---
    run_dir: Path

    # --- misc ---
    seed: int = 0
    device: str = "cuda"
    first_exp_epochs: int | None = None
    eval_every: int = -1
    checkpoint: bool = False
