from typing import Any, Callable, List, Tuple

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks.scenarios import NCScenario
from loguru import logger
from torchvision import transforms as T

from bayescl.config import Config
from bayescl.datasets import SplitCIFAR100, SplitDomainNet, SplitImageNetR

Transform = Callable[[Any], Any]

TRAIN_TRANSFORMS = {
    "SplitCIFAR100": [
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
    ],
    "SplitImageNetR": [
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
    ],
    "SplitDomainNet": [
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
    ],
    "CORe50": [
        T.RandomResizedCrop(128),
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
COMMON_POST_TRANSFORM = [
    T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
]


def get_transforms(cfg: Config, dataset: str) -> Tuple[Transform, Transform]:
    train_transform: List[Callable[[Any], Any]] = []
    eval_transform: List[Callable[[Any], Any]] = []

    train_transform.extend(COMMON_PRE_TRANSFORM)
    eval_transform.extend(COMMON_PRE_TRANSFORM)

    train_transform.extend(TRAIN_TRANSFORMS.get(dataset, []))
    eval_transform.extend(TEST_TRANSFORM)

    train_transform.extend(COMMON_POST_TRANSFORM)
    eval_transform.extend(COMMON_POST_TRANSFORM)

    return (T.Compose(train_transform), T.Compose(eval_transform))


def get_benchmark(cfg: Config) -> NCScenario:
    logger.info(f"Setting up '{cfg.scenario.dataset}' benchmark")
    train_transform, eval_transform = get_transforms(cfg, cfg.scenario.dataset)
    if cfg.scenario.dataset == "SplitMNIST":
        return SplitMNIST(
            dataset_root=cfg.dataset_root,
            n_experiences=cfg.scenario.n_tasks,
            # train_transform=train_transform,
            # eval_transform=eval_transform,
            return_task_id=True,
            shuffle=cfg.scenario.shuffle,
        )
    elif cfg.scenario.dataset == "SplitCIFAR100":
        return SplitCIFAR100(  # type: ignore
            dataset_root=cfg.dataset_root,
            n_experiences=cfg.scenario.n_tasks,
            train_transform=train_transform,
            eval_transform=eval_transform,
            return_task_id=True,
            shuffle=cfg.scenario.shuffle,
            validation_set=cfg.scenario.validation_set,
        )
    elif cfg.scenario.dataset == "SplitImageNetR":
        return SplitImageNetR(  # type: ignore
            dataset_root=cfg.dataset_root,
            n_experiences=cfg.scenario.n_tasks,
            train_transform=train_transform,
            eval_transform=eval_transform,
            return_task_id=True,
            shuffle=cfg.scenario.shuffle,
            validation_set=cfg.scenario.validation_set,
        )
    elif cfg.scenario.dataset == "SplitDomainNet":
        return SplitDomainNet(  # type: ignore
            dataset_root=cfg.dataset_root,
            n_experiences=cfg.scenario.n_tasks,
            train_transform=train_transform,
            eval_transform=eval_transform,
            return_task_id=True,
            shuffle=cfg.scenario.shuffle,
            validation_set=cfg.scenario.validation_set,
        )

    raise ValueError(f"Unsupported scenario: {cfg.scenario}")
