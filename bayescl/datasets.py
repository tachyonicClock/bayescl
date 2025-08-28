from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from avalanche.benchmarks import (
    CLScenario,
    nc_benchmark,
)
from claiutil.env import datasets_path
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder


class ImageNetR(ImageFolder):
    def __init__(
        self,
        root: str | Path,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
    ):
        super().__init__(Path(root) / "imagenet-r", transform, target_transform)


class DomainNet(Dataset):
    def __init__(
        self,
        root: str | Path,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        train: bool = True,
    ):
        self.root = Path(root) / "DomainNet"
        self.transform = transform
        self.target_transform = target_transform
        super().__init__()

        if train:
            with (self.root / "domainnet_train.yaml").open("r") as f:
                self._metadata = yaml.safe_load(f)
        else:
            with (self.root / "domainnet_test.yaml").open("r") as f:
                self._metadata = yaml.safe_load(f)

        self._data = self._metadata["data"]
        self.targets = torch.tensor(self._metadata["targets"], dtype=torch.int32)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        path = self.root / self._data[index]
        target = self.targets[index]
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, int(target)


def SplitImageNetR(
    dataset_root: str | Path = datasets_path(),
    n_experiences: int = 20,
    train_transform: Callable[..., Any] | None = None,
    eval_transform: Callable[..., Any] | None = None,
    seed: int | None = None,
    return_task_id: bool = False,
    shuffle: bool = True,
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
    # Create train/test splits
    randperm = torch.randperm(30_000, generator=torch.Generator().manual_seed(0)).int()
    test_perm, train_perm = randperm[:6_000], randperm[6_000:]
    train_dataset = Subset(ImageNetR(dataset_root, train_transform), train_perm)  # type: ignore
    test_dataset = Subset(ImageNetR(dataset_root, eval_transform), test_perm)  # type: ignore

    return nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
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
    # Create train/test splits
    train_dataset = DomainNet(dataset_root, train_transform, train=True)  # type: ignore
    test_dataset = DomainNet(dataset_root, eval_transform, train=False)  # type: ignore

    return nc_benchmark(
        train_dataset=train_dataset,  # type: ignore
        test_dataset=test_dataset,  # type: ignore
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        shuffle=shuffle,
    )
