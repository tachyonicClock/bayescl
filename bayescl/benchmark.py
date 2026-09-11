import random
from typing import TYPE_CHECKING, Any, Callable, List, Tuple

import numpy as np
import torch
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks.scenarios import NCScenario
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from bayescl.datasets import (
    SplitCIFAR100,
    SplitCLEAR10,
    SplitCORe50,
    SplitCUB200_2011,
    SplitDomainNet,
    SplitImageNetR,
)
from bayescl.metrics.corruptions import corrupt, corruption_names

if TYPE_CHECKING:
    from bayescl.config import ExperimentConfig
Transform = Callable[[Any], Any]

TRAIN_TRANSFORMS = {
    "CIFAR100": [
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
    ],
    "ImageNetR": [
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
    ],
    "DomainNet": [
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
    ],
    "CORe50": [
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
    ],
    "CLEAR10": [
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
    ],
    "CUB200_2011": [
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
    ],
    "TinyImageNet": [
        T.RandomResizedCrop(64),
        T.RandomHorizontalFlip(),
    ],
}
TEST_TRANSFORM = [
    T.Resize(256),
    T.CenterCrop(224),
]
# Tensor conversion runs *after* the geometric transforms above so that resizing /
# cropping happens on the (smaller, uint8) PIL image rather than on a full-size
# float tensor. On ImageNet-R this roughly doubles DataLoader throughput because
# ``RandomResizedCrop`` then only has to resample a 224px crop, not the whole
# multi-megapixel image.
COMMON_POST_TRANSFORM = [
    T.ToTensor(),
]

# ImageNet normalization values, commonly used for other datasets as well
STANDARDIZE = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def get_transforms(standardize: bool, dataset: str) -> Tuple[Transform, Transform]:
    train_transform: List[Callable[[Any], Any]] = []
    eval_transform: List[Callable[[Any], Any]] = []

    train_transform.extend(TRAIN_TRANSFORMS.get(dataset, []))
    eval_transform.extend(TEST_TRANSFORM)

    train_transform.extend(COMMON_POST_TRANSFORM)
    eval_transform.extend(COMMON_POST_TRANSFORM)

    if standardize:
        train_transform.append(STANDARDIZE)
        eval_transform.append(STANDARDIZE)

    return (T.Compose(train_transform), T.Compose(eval_transform))


def eval_transform_for(config: "ExperimentConfig") -> Transform:
    """The plain eval transform, e.g. to preprocess OOD datasets the same way
    as ``config.dataset``'s own eval split."""
    _, eval_transform = get_transforms(config.standardize, config.dataset)
    return eval_transform


class ShiftedTensorDataset(Dataset):
    """Wraps an already-preprocessed experience dataset and applies a random
    ImageNet-C-style corruption to each sample (EXPERIMENT.md ยง4.2, the
    ``$shift`` = ``severity`` eval data for ``ece@$shift``/``ace@$shift``).

    Every dataset construction path in :func:`get_benchmark` (``nc_benchmark``
    for CIFAR100/ImageNet-R, ``create_generic_benchmark_from_paths`` for
    CLEAR10) ends up producing ``(x, y, ...)`` items where ``x`` is already a
    fully preprocessed ``(standardize(ToTensor(CenterCrop(Resize(...))))``
    tensor -- but at a *different* point relative to Avalanche's own
    transform-group machinery in each case. Operating on the final tensor
    output, rather than trying to splice a corruption step into the
    upstream transform pipeline, sidesteps that and works identically for
    every dataset type: un-normalize back to a ``[0, 1]`` image, corrupt in
    pixel space, then re-normalize.
    """

    def __init__(self, base: Dataset, severity: int, standardize: bool) -> None:
        self.base = base
        self.severity = severity
        self.standardize = standardize
        self.targets = list(base.targets)  # type: ignore[attr-defined]
        mean = torch.tensor(STANDARDIZE.mean).view(-1, 1, 1)
        std = torch.tensor(STANDARDIZE.std).view(-1, 1, 1)
        self._mean, self._std = mean, std

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        # Only keep (x, y): ``base`` may already be an AvalancheDataset whose
        # items carry their own task label, and wrapping again through
        # ``create_multi_dataset_generic_benchmark`` adds a fresh one -- so
        # forwarding the original would leave two task-label fields in the
        # item tuple.
        item = self.base[idx]
        x, y = item[0], item[1]
        if self.standardize:
            x = x * self._std + self._mean
        arr = (x.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image = corrupt(Image.fromarray(arr), random.choice(corruption_names()), self.severity)
        out = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        if self.standardize:
            out = (out - self._mean) / self._std
        return out, y


def get_benchmark(config: "ExperimentConfig") -> NCScenario:
    logger.info(f"Setting up '{config.dataset}' benchmark")
    dataset_root = str(config.dataset_root)
    validation_set = 0.1 if config.validation else 0
    train_transform, eval_transform = get_transforms(config.standardize, config.dataset)
    if config.dataset == "MNIST":
        return SplitMNIST(
            dataset_root=dataset_root,
            n_experiences=config.n_tasks,
            # train_transform=train_transform,
            # eval_transform=eval_transform,
            return_task_id=True,
            shuffle=config.shuffle,
        )
    elif config.dataset == "CIFAR100":
        return SplitCIFAR100(  # type: ignore
            dataset_root=dataset_root,
            n_experiences=config.n_tasks,
            train_transform=train_transform,
            eval_transform=eval_transform,
            return_task_id=True,
            shuffle=config.shuffle,
            scale=config.scale,  # type: ignore
            validation=config.validation,
        )
    elif config.dataset == "ImageNetR":
        return SplitImageNetR(  # type: ignore
            dataset_root=dataset_root,
            n_experiences=config.n_tasks,
            train_transform=train_transform,
            eval_transform=eval_transform,
            return_task_id=True,
            shuffle=config.shuffle,
            scale=config.scale,  # type: ignore
            validation=config.validation,
        )
    elif config.dataset == "CLEAR10":
        # Domain-incremental: buckets are fixed native experiences, no
        # n_experiences/shuffle knob like the class-incremental Split* above.
        return SplitCLEAR10(  # type: ignore
            dataset_root=dataset_root,
            train_transform=train_transform,
            eval_transform=eval_transform,
            return_task_id=True,
            scale=config.scale,  # type: ignore
            validation=config.validation,
        )
    elif config.dataset == "DomainNet":
        return SplitDomainNet(  # type: ignore
            dataset_root=dataset_root,
            n_experiences=config.n_tasks,
            train_transform=train_transform,
            eval_transform=eval_transform,
            return_task_id=True,
            shuffle=config.shuffle,
            validation_set=validation_set,
        )
    elif config.dataset == "CORe50":
        # SplitCORe50 takes a bool here, every other Split* takes a float fraction.
        return SplitCORe50(  # type: ignore
            dataset_root=dataset_root,
            n_experiences=config.n_tasks,
            train_transform=train_transform,
            eval_transform=eval_transform,
            return_task_id=True,
            shuffle=config.shuffle,
            validation_set=config.validation,
        )
    elif config.dataset == "CUB200_2011":
        return SplitCUB200_2011(  # type: ignore
            dataset_root=dataset_root,
            n_experiences=config.n_tasks,
            train_transform=train_transform,
            eval_transform=eval_transform,
            return_task_id=True,
            shuffle=config.shuffle,
            validation_set=validation_set,
        )
    raise ValueError(f"Unsupported scenario: {config.dataset}")
