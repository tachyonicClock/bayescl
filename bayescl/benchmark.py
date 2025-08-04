from typing import Any, Callable, List, Tuple

from avalanche.benchmarks.classic import CORe50, SplitCIFAR10, SplitCIFAR100, SplitMNIST
from avalanche.benchmarks.scenarios import NCScenario
from loguru import logger
from torchvision import transforms as T
from transformers import AutoImageProcessor

from bayescl.config import Config
from bayescl.datasets import SplitImageNetR

Transform = Callable[[Any], Any]

LUT_TRAIN_TRANSFORMS = {
    "SplitCIFAR100": [
        T.RandomCrop(32, 4),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
    ],
    "SplitImageNetR": [
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
    ],
    "CORe50": [
        T.RandomResizedCrop(128),
        T.RandomHorizontalFlip(),
    ],
}


def get_transforms(cfg: Config) -> Tuple[Transform, Transform]:
    train_transform: List[Callable[[Any], Any]] = [T.ToTensor()]
    train_transform.extend(LUT_TRAIN_TRANSFORMS.get(cfg.scenario.dataset, []))
    eval_transform: List[Callable[[Any], Any]] = [T.ToTensor()]

    if cfg.model.type == "huggingface":
        logger.info(
            "huggingface model uses AutoImageProcessor for pre-processing transforms."
        )
        pre_process = AutoImageProcessor.from_pretrained(cfg.model.name, use_fast=True)
        normalize = T.Normalize(mean=pre_process.image_mean, std=pre_process.image_std)
        resize = T.Resize(pre_process.size["shortest_edge"])
        crop = T.CenterCrop(
            (pre_process.crop_size["height"], pre_process.crop_size["width"])
        )
        train_transform.extend([normalize, resize, crop])
        eval_transform.extend([normalize, resize, crop])

    return (
        T.Compose(train_transform),
        T.Compose(eval_transform),
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
