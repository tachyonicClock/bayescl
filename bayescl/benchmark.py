from typing import TYPE_CHECKING, Any, Callable, List, Tuple

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks.scenarios import NCScenario
from loguru import logger
from torchvision import transforms as T

from bayescl.datasets import (
    SplitCIFAR100,
    SplitCORe50,
    SplitCUB200_2011,
    SplitDomainNet,
    SplitImageNetR,
)
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
COMMON_PRE_TRANSFORM = [
    T.ToTensor(),
]
COMMON_POST_TRANSFORM = []

# ImageNet normalization values, commonly used for other datasets as well
STANDARDIZE = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def get_transforms(standardize: bool, dataset: str) -> Tuple[Transform, Transform]:
    train_transform: List[Callable[[Any], Any]] = []
    eval_transform: List[Callable[[Any], Any]] = []

    train_transform.extend(COMMON_PRE_TRANSFORM)
    eval_transform.extend(COMMON_PRE_TRANSFORM)

    train_transform.extend(TRAIN_TRANSFORMS.get(dataset, []))
    eval_transform.extend(TEST_TRANSFORM)

    train_transform.extend(COMMON_POST_TRANSFORM)
    eval_transform.extend(COMMON_POST_TRANSFORM)

    if standardize:
        train_transform.append(STANDARDIZE)
        eval_transform.append(STANDARDIZE)

    return (T.Compose(train_transform), T.Compose(eval_transform))


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
            validation_set=validation_set,
        )
    elif config.dataset == "ImageNetR":
        return SplitImageNetR(  # type: ignore
            dataset_root=dataset_root,
            n_experiences=config.n_tasks,
            train_transform=train_transform,
            eval_transform=eval_transform,
            return_task_id=True,
            shuffle=config.shuffle,
            validation_set=validation_set,
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
