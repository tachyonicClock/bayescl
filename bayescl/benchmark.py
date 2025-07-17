from avalanche.benchmarks.classic import CORe50, SplitCIFAR10, SplitMNIST
from avalanche.benchmarks.scenarios import NCScenario
from loguru import logger

from bayescl.config import Config


def get_benchmark(cfg: Config) -> NCScenario:
    train_transform = None
    eval_transform = None

    if cfg.model.type == "huggingface":
        from transformers import AutoImageProcessor

        logger.info(
            "huggingface model uses AutoImageProcessor for pre-processing transforms."
        )
        image_processor = AutoImageProcessor.from_pretrained(cfg.model.name)

        def train_transform(img):
            return image_processor(img, return_tensors="pt").pixel_values.squeeze(0)

        def eval_transform(img):
            return image_processor(img, return_tensors="pt").pixel_values.squeeze(0)

    if cfg.scenario.dataset == "SplitMNIST":
        return SplitMNIST(
            dataset_root=cfg.dataset_root,
            n_experiences=cfg.scenario.n_tasks,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
    elif cfg.scenario.dataset == "SplitCIFAR10":
        return SplitCIFAR10(
            dataset_root=cfg.dataset_root,
            n_experiences=cfg.scenario.n_tasks,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
    elif cfg.scenario.dataset == "CORe50":
        return CORe50(  # type: ignore
            dataset_root=cfg.dataset_root,
            run=cfg.scenario.run,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
    else:
        raise ValueError(f"Unsupported scenario: {cfg.scenario}")
