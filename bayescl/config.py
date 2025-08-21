from os import environ
from pathlib import Path
from typing import Literal, Optional

import torch
from claiutil.optuna import HyperparameterSearch
from claiutil.peft import BLoBConfig
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class BaseConfig(BaseModel):
    model_config: ConfigDict = {"extra": "forbid"}  # type: ignore


# --- Scenario Configurations ---


class Scenario(BaseConfig):
    dataset: Literal[
        "SplitMNIST", "SplitCIFAR10", "SplitCIFAR100", "SplitImageNetR"
    ] = "SplitMNIST"
    n_tasks: int = 5
    shuffle: bool = True


class ScenarioCORe50(BaseConfig):
    dataset: Literal["CORe50"] = "CORe50"
    scenario: Literal["nc", "ni", "nicv2_79"] = "nc"
    run: int = 0


# --- Model Configurations ---


class BasicModelConfig(BaseConfig):
    type: Literal["basic"] = "basic"
    name: str = "SimpleMLP"


class HuggingFaceModelConfig(BaseConfig):
    type: Literal["huggingface"] = "huggingface"
    name: str = "facebook/dinov2-small"
    freeze_backbone: bool = True


# --- Plugin Configurations ---


# --- PEFT Configurations ---


class LoRAConfig(BaseConfig):
    type: Literal["LoRA"] = "LoRA"
    r: int = 16
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    head_module: str = "model.classifier"


class CLoRAConfig(BaseConfig):
    type: Literal["CLoRA"] = "CLoRA"
    r: int = 4
    lambda_: float = 1.0
    head_module: str = "model.classifier"


class InfLoRAConfig(BaseConfig):
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
    head_module: str = "model.classifier"


class BLoB(BaseConfig):
    """BLoB: Bayesian low-rank adaptation by backpropagation for large language
    models
    """

    type: Literal["BLoB"] = "BLoB"
    head_module: str = "model.classifier"
    #: strength of the kl divergence loss
    beta: float = 1.0
    #: number of samples to use for bayesian evaluation
    bayes_eval_samples: int = 0
    config: BLoBConfig


class Config(BaseConfig):
    include: list[Path] = []
    label: str = "default"

    #: How many times should the experiment be run with this config
    repeat: int = 1

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
    #: Number of workers for data loading
    num_workers: int = 0
    #: Number of samples in the replay memory
    replay: int = 0

    #: Stop after training on this many tasks
    max_tasks: Optional[int] = None

    #: Use local cross entropy by masking the output layer during training
    use_local_ce: bool = True

    hpsearch: Optional[HyperparameterSearch] = None


def _resolve_includes(base: Path, filenames: list[str]) -> DictConfig:
    if filenames:
        return OmegaConf.merge(*(OmegaConf.load(base / f) for f in filenames))  # type: ignore
    else:
        return DictConfig({})


def from_configs(
    config_filenames: list[str], dotlist: list[str] | None = None
) -> Config:
    config = DictConfig({})  # type: ignore
    for file in config_filenames:
        file = Path(file)
        config = OmegaConf.merge(config, OmegaConf.load(file))  # type: ignore
        included = _resolve_includes(file.parent, config.get("include", []))  # type: ignore
        config = OmegaConf.merge(included, config)
    if dotlist is not None:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(dotlist))

    return Config.model_validate(OmegaConf.to_object(config))
