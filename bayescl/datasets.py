from pathlib import Path
from typing import Any, Callable

import torch
from avalanche.benchmarks import (
    CLScenario,
    nc_benchmark,
)
from claiutil.env import datasets_path
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder


class ImageNetR(ImageFolder):
    def __init__(
        self,
        root: str | Path,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
    ):
        super().__init__(Path(root) / "imagenet-r", transform, target_transform)


def SplitImageNetR(
    dataset_root: str | Path = datasets_path(),
    n_experiences: int = 20,
    train_transform: Callable[..., Any] | None = None,
    eval_transform: Callable[..., Any] | None = None,
    seed: int = 0,
    return_task_id: bool = False,
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
    )
