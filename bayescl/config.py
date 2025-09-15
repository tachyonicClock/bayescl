from os import environ
from pathlib import Path
from typing import Literal, Optional

import torch
from claiutil.optuna import HyperparameterSearch
from claiutil.peft import BLoBConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class BaseConfig(BaseModel):
    model_config: ConfigDict = {"extra": "forbid"}  # type: ignore


# --- Scenario Configurations ---


class Scenario(BaseConfig):
    dataset: Literal[
        "SplitMNIST",
        "SplitCIFAR10",
        "SplitCIFAR100",
        "SplitImageNetR",
        "SplitDomainNet",
    ] = "SplitMNIST"
    n_tasks: int = 5
    shuffle: bool = True
    #: Replace the test set with a validation set of this size (0.0 to 1.0)
    validation_set: float = 0.0


class ScenarioCORe50(BaseConfig):
    dataset: Literal["CORe50"] = "CORe50"
    scenario: Literal["nc", "ni", "nicv2_79"] = "nc"
    run: int = 0


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


class BLoB(PEFTConfig):
    """BLoB: Bayesian low-rank adaptation by backpropagation for large language
    models
    """

    type: Literal["BLoB"] = "BLoB"
    #: strength of the kl divergence loss
    beta: float = 1.0
    #: number of samples to use for bayesian evaluation
    bayes_eval_samples: int = 0

    #: Use a variational bayesian last layer?
    vbll: bool = False

    config: BLoBConfig


class Config(BaseConfig):
    include: list[Path] = []
    label: str = "default"

    #: Scenario configuration.
    scenario: Scenario | ScenarioCORe50 = Field(
        Scenario(),
        discriminator="dataset",
    )
    #: Model configuration.
    model: BasicModelConfig | HuggingFaceModelConfig = Field(
        BasicModelConfig(),
        discriminator="type",
    )

    peft: Optional[LoRAConfig | BLoB | CLoRAConfig | InfLoRAConfig] = Field(
        None,
        discriminator="type",
    )

    #: Parent directory containing datasets.
    dataset_root: str = environ.get("DATASETS", "./datasets")
    #: Parent directory containing run logs.
    log_root: str = "./log"
    study_name: str = "manual"

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
    #: Number of epochs in the first experience
    first_exp_epochs: Optional[int] = None

    #: Add the outlier exposure plugin
    outlier_exposure: Optional[OutlierExposureConfig] = None

    #: Number of workers for data loading
    num_workers: int = 0
    #: Number of samples in the replay memory
    replay: int = 0

    #: Stop after training on this many tasks
    max_tasks: Optional[int] = None

    #: Use local cross entropy by masking the output layer during training
    use_local_ce: bool = True

    hpsearch: Optional[HyperparameterSearch] = None

    #: When using hyperparameter search, scale the number of epochs by this factor
    hpsearch_epoch_scale: Optional[float] = None

    run_id: Optional[str] = None


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
