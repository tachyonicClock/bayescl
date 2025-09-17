import os
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import torch
import yaml
from avalanche.benchmarks import (
    CLScenario,
    nc_benchmark,
)
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR100, ImageFolder


def datasets_path() -> str:
    """Get the root directory for PyTorch."""
    torch_data_dir_ = os.environ.get("DATASETS")
    if not torch_data_dir_:
        raise RuntimeError("The ``DATASETS`` environment variable should be set.")
    torch_data_dir_path = Path(torch_data_dir_).expanduser().resolve()
    torch_data_dir_path.mkdir(exist_ok=True)
    return str(torch_data_dir_path)


class ImageNetR(ImageFolder):
    def __init__(
        self,
        root: str | Path,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        train: bool = True,
    ):
        path = Path(root) / "imagenet-r" / ("train" if train else "test")
        super().__init__(path, transform, target_transform)


class DomainNet(Dataset):
    def __init__(
        self,
        root: str | Path,
        transform: Callable[..., Any] | None = None,
        train: bool = True,
    ):
        self.root = Path(root) / "domainnet"
        self.transform = transform
        super().__init__()

        if train:
            with (self.root / "domainnet_train.yaml").open("r") as f:
                self._metadata = yaml.safe_load(f)
        else:
            with (self.root / "domainnet_test.yaml").open("r") as f:
                self._metadata = yaml.safe_load(f)

        self._data = self._metadata["data"]
        self.targets = self._metadata["targets"]
        self.classes = list(range(345))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        path = self.root / self._data[index]
        target = self.targets[index]
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, int(target)


class TinyImageNet(ImageFolder):
    def __init__(
        self,
        root: str | Path,
        transform: Callable[..., Any] | None = None,
        split: Literal["train", "val", "test"] = "train",
    ):
        super().__init__(Path(root) / "tiny-imagenet-200" / split, transform)


def valid_split_indices(
    n: int, validation_set: float
) -> tuple[Sequence[int], Sequence[int]]:
    assert 0.0 < validation_set < 1.0
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(0)).int()
    n_valid = int(n * validation_set)
    train_indices, valid_indices = indices[n_valid:], indices[:n_valid]
    logger.info(
        f"Splitting {n} samples into {len(train_indices)} train and {len(valid_indices)} valid"
    )
    return train_indices, valid_indices  # type: ignore


def SplitImageNetR(
    dataset_root: str | Path = datasets_path(),
    n_experiences: int = 20,
    train_transform: Callable[..., Any] | None = None,
    eval_transform: Callable[..., Any] | None = None,
    seed: int | None = None,
    return_task_id: bool = False,
    shuffle: bool = True,
    validation_set: float = 0.0,
) -> CLScenario:
    """Create SplitImageNetR200 by splitting ImageNet-R(endition)[#f1].

    >>> benchmark = SplitImageNetR()
    >>> for experience in benchmark.train_stream:
    ...     print(experience.current_experience, experience.classes_in_this_experience)
    0 [64, 65, 74, 10, 44, 81, 19, 118, 57, 157]
    1 [128, 193, 134, 73, 141, 116, 84, 149, 24, 61]
    2 [98, 163, 38, 199, 75, 11, 46, 15, 80, 63]
    3 [35, 36, 8, 43, 109, 50, 21, 54, 55, 29]
    4 [70, 7, 72, 41, 170, 177, 146, 87, 58, 159]
    5 [97, 165, 123, 78, 113, 151, 90, 155, 93, 62]
    6 [1, 136, 138, 48, 188, 119, 91, 60, 190, 31]
    ...

    .. [#f1] Hendrycks, Dan, et al. "The many faces of robustness: A critical analysis
        of out-of-distribution generalization." Proceedings of the IEEE/CVF
        international conference on computer vision. 2021.
    """
    n = 24000
    if validation_set <= 0.0:
        train_dataset = ImageNetR(dataset_root, train_transform, train=True)
        test_dataset = ImageNetR(dataset_root, eval_transform, train=False)
        assert len(train_dataset) == n
    else:
        train_perm, test_perm = valid_split_indices(n, validation_set)
        train_dataset = Subset(
            ImageNetR(dataset_root, train_transform, train=True), train_perm
        )  # type: ignore
        test_dataset = Subset(
            ImageNetR(dataset_root, eval_transform, train=True), test_perm
        )  # type: ignore

    return nc_benchmark(
        train_dataset=train_dataset,  # type: ignore
        test_dataset=test_dataset,  # type: ignore
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        shuffle=shuffle,
    )


def SplitDomainNet(
    dataset_root: str | Path = datasets_path(),
    n_experiences: int = 5,
    train_transform: Callable[..., Any] | None = None,
    eval_transform: Callable[..., Any] | None = None,
    seed: int | None = None,
    return_task_id: bool = False,
    shuffle: bool = True,
    validation_set: float = 0.0,
) -> CLScenario:
    """Create SplitDomainNet by splitting DomainNet[#f1].

    >>> benchmark = SplitDomainNet()
    >>> for experience in benchmark.train_stream:
    ...     print(experience.current_experience, len(experience.classes_in_this_experience))
    0 69
    1 69
    2 69
    3 69
    4 69

    .. [#f1] Peng, Xialei, et al. "Moment matching for multi-source domain
        adaptation." Proceedings of the IEEE/CVF International Conference on
        Computer Vision. 2019.
    """
    n = 338804
    if validation_set <= 0.0:
        train_dataset = DomainNet(dataset_root, train=True, transform=train_transform)  # type: ignore
        test_dataset = DomainNet(dataset_root, train=False, transform=eval_transform)  # type: ignore
        assert len(train_dataset) == n
    else:
        train_perm, test_perm = valid_split_indices(n, validation_set)
        train_dataset = Subset(
            DomainNet(dataset_root, train=True, transform=train_transform), train_perm
        )
        test_dataset = Subset(
            DomainNet(dataset_root, train=True, transform=eval_transform), test_perm
        )

    return nc_benchmark(
        train_dataset=train_dataset,  # type: ignore
        test_dataset=test_dataset,  # type: ignore
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        shuffle=shuffle,
    )


def SplitCIFAR100(
    dataset_root: str | Path = datasets_path(),
    n_experiences: int = 5,
    train_transform: Callable[..., Any] | None = None,
    eval_transform: Callable[..., Any] | None = None,
    seed: int | None = None,
    return_task_id: bool = False,
    shuffle: bool = True,
    validation_set: float = 0.0,
) -> CLScenario:
    # Create train/test splits
    n = 50000
    if validation_set <= 0.0:
        train_dataset = CIFAR100(dataset_root, train=True, transform=train_transform)  # type: ignore
        test_dataset = CIFAR100(dataset_root, train=False, transform=eval_transform)  # type: ignore
        assert len(train_dataset) == n
    else:
        train_perm, test_perm = valid_split_indices(n, validation_set)
        train_dataset = Subset(
            CIFAR100(dataset_root, train=True, transform=train_transform), train_perm
        )
        test_dataset = Subset(
            CIFAR100(dataset_root, train=True, transform=eval_transform), test_perm
        )

    return nc_benchmark(
        train_dataset=train_dataset,  # type: ignore
        test_dataset=test_dataset,  # type: ignore
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        shuffle=shuffle,
    )
