from typing import Any, Callable, Tuple

from avalanche.benchmarks.classic import CORe50, SplitCIFAR10, SplitCIFAR100, SplitMNIST
from avalanche.benchmarks.scenarios import NCScenario
from loguru import logger
from torch.nn import Identity
from torchvision import transforms as T
from transformers import AutoImageProcessor

from bayescl.config import Config
from bayescl.datasets import SplitImageNetR

Transform = Callable[[Any], Any]

LUT_TRAIN_TRANSFORMS = {
    "SplitCIFAR100": T.Compose(
        [
            T.RandomCrop(32, 4),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
        ]
    ),
    "SplitImageNetR": T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
        ]
    ),
    "CORe50": T.Compose(
        [
            T.RandomResizedCrop(128),
            T.RandomHorizontalFlip(),
        ]
    ),
}


def get_transforms(cfg: Config) -> Tuple[Transform, Transform]:
    identity_fn = Identity()
    normalize: Callable[[Any], Any] = identity_fn
    resize: Callable[[Any], Any] = identity_fn

    if cfg.model.type == "huggingface":
        logger.info(
            "huggingface model uses AutoImageProcessor for pre-processing transforms."
        )
        pre_process = AutoImageProcessor.from_pretrained(cfg.model.name, use_fast=True)
        normalize = T.Normalize(mean=pre_process.image_mean, std=pre_process.image_std)
        resize = T.Resize(pre_process.size["shortest_edge"])

    train_transform = LUT_TRAIN_TRANSFORMS.get(cfg.scenario.dataset, identity_fn)
    if train_transform is identity_fn:
        logger.warning("No training transforms, consider adding some.")

    return (
        T.Compose([T.ToTensor(), train_transform, normalize, resize]),
        T.Compose([T.ToTensor(), normalize, resize]),
    )


def get_benchmark(cfg: Config) -> NCScenario:
    train_transform, eval_transform = get_transforms(cfg)
    if cfg.scenario.dataset == "SplitMNIST":
        return SplitMNIST(
            dataset_root=cfg.dataset_root,
            n_experiences=cfg.scenario.n_tasks,
            # train_transform=train_transform,
            # eval_transform=eval_transform,
            return_task_id=True,
        )
    elif cfg.scenario.dataset == "SplitCIFAR10":
        return SplitCIFAR10(
            dataset_root=cfg.dataset_root,
            n_experiences=cfg.scenario.n_tasks,
            train_transform=train_transform,
            eval_transform=eval_transform,
            return_task_id=True,
        )
    elif cfg.scenario.dataset == "SplitCIFAR100":
        return SplitCIFAR100(
            dataset_root=cfg.dataset_root,
            n_experiences=cfg.scenario.n_tasks,
            train_transform=train_transform,
            eval_transform=eval_transform,
            return_task_id=True,
        )
    elif cfg.scenario.dataset == "SplitImageNetR":
        return SplitImageNetR(  # type: ignore
            dataset_root=cfg.dataset_root,
            n_experiences=cfg.scenario.n_tasks,
            train_transform=train_transform,
            eval_transform=eval_transform,
            return_task_id=True,
        )
    elif cfg.scenario.dataset == "CORe50":
        return CORe50(  # type: ignore
            dataset_root=cfg.dataset_root,
            scenario=cfg.scenario.scenario,
            run=cfg.scenario.run,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
    else:
        raise ValueError(f"Unsupported scenario: {cfg.scenario}")
