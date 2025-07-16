from os import environ
from typing import Literal, Optional

import torch
from pydantic import BaseModel, ConfigDict, Field


class BaseConfig(BaseModel):
    model_config: ConfigDict = {"extra": "forbid"}  # type: ignore


class Scenario(BaseConfig):
    dataset: Literal["SplitMNIST"] = "SplitMNIST"
    n_tasks: int = 5


class ScenarioCORe50(BaseConfig):
    dataset: Literal["CORe50"] = "CORe50"
    scenario: Literal["nc"] = "nc"
    run: int = 0


class Config(BaseConfig):
    # Scenario Configuration
    scenario: Scenario = Field(
        Scenario(),
        discriminator="dataset",
    )

    #: Parent directory containing datasets.
    dataset_root: str = environ.get("DATASETS", "./datasets")
    #: Parent directory containing run logs.
    log_root: str = "./log"

    # Strategy
    #: Device to use for training (cuda or cpu)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    #: Mini-batch size for training
    train_mb_size: int = 500
    #: Mini-batch size for evaluation. If None, defaults to train_mb_size
    eval_mb_size: Optional[int] = None
    #: Number of epochs for training each experience
    train_epochs: int = 1
    #: Number of workers for data loading
    num_workers: int = 4
