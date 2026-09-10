"""Dataset definitions.

Replaces the ``configs/base/dataset/*.jsonnet`` layer: the backbone, adapter
filter, batch sizes, task count and per-dataset epoch budget all live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Backbone:
    name: str = "microsoft/resnet-18"
    freeze_backbone: bool = True
    #: Regex selecting which submodules receive adapters.
    adapter_filter: str = r".*layer\.[0-9]\.convolution"
    head_module: str = "model.classifier.1"


@dataclass(frozen=True)
class Dataset:
    key: str
    #: Avalanche scenario name consumed by :func:`bayescl.benchmark.get_benchmark`.
    scenario: str
    n_tasks: int = 10
    shuffle: bool = False
    #: Training epochs per experience for the ``full`` scale.
    full_epochs: int = 30
    train_mb_size: int = 128
    eval_mb_size: int = 256
    num_workers: int = 4
    standardize: bool = True
    backbone: Backbone = field(default_factory=Backbone)


DATASETS: dict[str, Dataset] = {
    "cifar100": Dataset("cifar100", "CIFAR100", full_epochs=30),
    "core50": Dataset("core50", "CORe50", full_epochs=30),
    "imagenetr": Dataset("imagenetr", "ImageNetR", full_epochs=60),
}


def dataset_names() -> list[str]:
    return list(DATASETS)


def get_dataset(key: str) -> Dataset:
    return DATASETS[key]
