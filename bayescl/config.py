from os import environ
from pathlib import Path
from typing import Literal, Optional, Union

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from bayescl.peft import BALLConfig
from bayescl.util.optuna import HyperparameterSearch


class BaseConfig(BaseModel):
    model_config: ConfigDict = {"extra": "forbid"}  # type: ignore


# --- Scenario Configurations ---


class Scenario(BaseConfig):
    dataset: Literal[
        "MNIST",
        "CIFAR10",
        "CIFAR100",
        "ImageNetR",
        "DomainNet",
    ] = "MNIST"
    n_tasks: int = 5
    shuffle: bool = True
    #: Replace the test set with a validation set
    validation: bool = False


# --- Model Configurations ---


class BasicModelConfig(BaseConfig):
    type: Literal["basic"] = "basic"
    name: str = "SimpleMLP"
    #: If adapters are used, regex to filter which modules to add adapters to.
    adapter_module_filter: str = ""


class HuggingFaceModelConfig(BaseConfig):
    type: Literal["huggingface"] = "huggingface"
    name: str = "facebook/dinov2-small"
    freeze_backbone: bool = True
    #: If adapters are used, regex to filter which modules to add adapters to.
    adapter_filter: str = ".*(key|query)"


# --- Plugin Configurations ---


class OutlierExposureConfig(BaseModel):
    strength: float = 1.0
    batch_size: Optional[int] = None


class RWalkConfig(BaseModel):
    ewc_lambda: float
    """hyperparameter to weigh the penalty inside the total loss. The larger the lambda,
    the larger the regularization."""
    ewc_alpha: float
    """Specify the moving average factor for the importance matrix, as defined RWalk
    paper (a.k.a. EWC++). Higher values lead to higher weight to newly computed
    importances. Must be in [0, 1]. Defaults to 0.9.
    """
    delta_t: int
    """Specify the iterations interval in which the parameter scores are updated."""


# --- PEFT Configurations ---


class PEFTConfig(BaseModel):
    #: Name of the module containing the classification head
    head_module: str = "model.classifier"
    #: Optional checkpoint to load adapter weights from
    checkpoint: Optional[Path] = None
    #: Whether to save the adapter weights after training
    save: bool = False


class LoRAConfig(PEFTConfig):
    type: Literal["LoRA"] = "LoRA"
    r: int = 16
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    head_module: str = "model.classifier"


class CLoRAConfig(PEFTConfig):
    type: Literal["CLoRA"] = "CLoRA"
    r: int = 4
    lambda_: float = 1.0
    head_module: str = "model.classifier"


class InfLoRAConfig(PEFTConfig):
    """InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning

    Liang, Y.-S., & Li, W.-J. (2024). InfLoRA: Interference-Free Low-Rank Adaptation for
    Continual Learning. 23638–23647.
    """

    type: Literal["InfLoRA"] = "InfLoRA"
    r: int = 4
    threshold: float = 0.95
    """Also called epsilon in the paper. Controls how accurate the k-rank approximation
    of the representation is. Default threshold is 0.95, see Table 1 in Liang & Li (2024).
    """


class LinearAnneal(BaseModel):
    type: Literal["linear"] = "linear"
    start: float = 0.0
    """The position (as a fraction of task training) to start annealing."""
    end: float = 1.0
    """The position (as a fraction of task training) to end annealing."""


class CyclicAnneal(BaseModel):
    type: Literal["cyclic"] = "cyclic"
    cycles: float
    """Number of cycles per epoch"""


class BALL(PEFTConfig):
    type: Literal["BALL"] = "BALL"
    beta: float = 1.0
    """Hyperparameter weighting the KL divergence loss."""

    anneal: Union[None, LinearAnneal, CyclicAnneal] = Field(None, discriminator="type")
    """If specified, use annealing for beta over the course of training."""

    vbll: bool
    """If True, enable Variational Bayesian Last Layer"""

    train_samples: int
    """Number of samples for each step of training."""

    test_samples: int
    """Number of samples for each step of testing."""

    softmax_avg: bool = True
    """If true apply softmax before averaging the logits over samples.
    
    This is theoretically more sound but empirically tends to perform worse.
    """
    config: BALLConfig


# --- Strategy Configurations ---


class DERConfig(PEFTConfig):
    type: Literal["DER"] = "DER"
    #: Number of samples in the replay memory
    mem_size: int
    #: Hyperparameter weighting the MSE loss
    alpha: float
    #: Hyperparameter weighting the CE loss, when more than 0, DER++ is used instead of DER
    beta: float


class GDumbConfig(PEFTConfig):
    type: Literal["GDumb"] = "GDumb"
    #: Number of samples in the replay memory
    mem_size: int


class Label(BaseConfig):
    """Label given to the experiment: ``{study}/{scenario}/{method}/{run}``"""

    study: str = "manual"
    """Top level label for a group of related experiments."""
    scenario: str = "unlabeled_scenario"
    """Label for the scenario or dataset used."""
    method: str = "unlabeled_method"
    """Label for the method or strategy used."""
    run: Optional[str] = None
    """Label for the individual run, e.g. 0001 for hyperparameter search trials."""


class Config(BaseConfig):
    include: list[Path] = []

    label: Label = Field(Label())

    #: Scenario configuration.
    scenario: Scenario = Field(Scenario())

    #: Model configuration.
    model: BasicModelConfig | HuggingFaceModelConfig = Field(
        BasicModelConfig(),
        discriminator="type",
    )

    peft: Optional[LoRAConfig | BALL | CLoRAConfig | InfLoRAConfig] = Field(
        None,
        discriminator="type",
    )

    #: Parent directory containing datasets.
    dataset_root: str = environ.get("DATASETS", "./datasets")
    #: Parent directory containing run logs.
    log_root: str = "./log"

    # Strategy
    #: Device to use for training (cuda or cpu)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    #: Learning rate for the optimizer
    lr: float = 0.001
    #: Mini-batch size for training
    train_mb_size: int = 500
    #: Mini-batch size for evaluation. If None, defaults to train_mb_size
    eval_mb_size: Optional[int] = None
    #: Number of epochs for training each experience
    epochs: int = 1
    #: Evaluate every N epochs. If -1, only evaluate at the end of each experience
    eval_every: int = -1

    #: Number of epochs in the first experience
    first_exp_epochs: Optional[int] = None

    #: Add the outlier exposure plugin
    outlier_exposure: Optional[OutlierExposureConfig] = None

    #: Add the RWalk plugin
    rwalk: Optional[RWalkConfig] = None

    #: Number of workers for data loading
    num_workers: int = 0

    strategy: Optional[DERConfig | GDumbConfig] = Field(None, discriminator="type")
    """If None use naive strategy with configured plugins. If specified, use the given
    strategy with the configured plugins.
    """

    #: Number of samples in the replay memory
    replay: int = 0

    #: Stop after training on this many tasks
    max_tasks: Optional[int] = None

    #: Use local cross entropy by masking the output layer during training
    use_local_ce: bool = True

    hpsearch: Optional[HyperparameterSearch] = None

    #: Random seed for reproducibility
    seed: int = 0

    save_logits: bool = False
    """If True, save the logits and true labels after each evaluation to the results
    file. This can be used for further analysis, e.g. computing calibration metrics.
    """


def _resolve_includes(base: Path, filenames: list[str]) -> DictConfig:
    if filenames:
        logger.info(f"Including configs {filenames}")
        config = OmegaConf.merge(*(OmegaConf.load(base / f) for f in filenames))
        assert config.get("include") is None, "Nested includes are not supported"
        return config  # type: ignore
    else:
        return DictConfig({})


def from_configs(
    config_filenames: list[str], dotlist: list[str] | None = None
) -> Config:
    config = DictConfig({})  # type: ignore
    for file in config_filenames:
        file = Path(file)
        logger.info(f"Updating config from {file}")
        config = OmegaConf.merge(config, OmegaConf.load(file))  # type: ignore
        included = _resolve_includes(file.parent, config.get("include", []))  # type: ignore
        config = OmegaConf.merge(included, config)
    if dotlist is not None:
        logger.info(f"Updating config from command line args: {dotlist}")
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(dotlist))

    return Config.model_validate(OmegaConf.to_object(config))
