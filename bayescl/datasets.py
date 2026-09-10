import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

# Keep every Hugging Face artifact (hub blobs + processed Arrow) under ``$DATASETS``
# instead of the default ``~/.cache/huggingface``. Must run before importing
# ``datasets``/``huggingface_hub`` since the cache paths are resolved at import.
if os.environ.get("DATASETS") and not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = str(
        Path(os.environ["DATASETS"]).expanduser().resolve() / "huggingface"
    )

import torch
import yaml
from avalanche.benchmarks import (
    CLScenario,
    nc_benchmark,
)

from datasets import load_dataset
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


def hf_cache_dir() -> str:
    """Directory under ``$DATASETS`` where Hugging Face datasets are cached."""
    path = Path(datasets_path()) / "huggingface"
    path.mkdir(exist_ok=True)
    return str(path)


@lru_cache(maxsize=None)
def _load_hf(repo: str, split: str):
    """Load and cache a Hugging Face dataset split for the lifetime of the process.

    ``Split*`` helpers instantiate each dataset class two or three times (a length
    probe plus the train and test views), so without this cache every call would
    re-open the Arrow table and re-run the class-balanced split. The cache is
    populated in the parent process before ``DataLoader`` workers fork, so workers
    inherit it for free; the underlying Arrow data is memory-mapped, not copied.
    """
    return load_dataset(repo, split=split, cache_dir=hf_cache_dir())


@lru_cache(maxsize=None)
def _imagenetr_split() -> tuple[Any, list[str], dict[str, int], list[int], list[int], list[int]]:
    ds = _load_hf(ImageNetR.HF_REPO, "test")
    classes = sorted(set(ds["wnid"]))
    wnid_to_idx = {wnid: i for i, wnid in enumerate(classes)}
    all_targets = [wnid_to_idx[wnid] for wnid in ds["wnid"]]
    train_idx, test_idx = class_balanced_split(all_targets, 0.2, seed=0)
    return (
        ds,
        classes,
        wnid_to_idx,
        all_targets,
        [int(i) for i in train_idx],
        [int(i) for i in test_idx],
    )


class ImageNetR(Dataset):
    """ImageNet-R(endition) hosted on the Hugging Face Hub (``axiong/imagenet-r``).

    The Hub dataset ships a single ``test`` split of 30,000 images labelled by
    WordNet id. Integer labels are derived from the sorted list of WordNet ids and
    a deterministic class-balanced 80/20 split yields ~24,000 train / ~6,000 test
    images (the per-class counts are not exactly equal, so the split is not a clean
    150/30).
    """

    HF_REPO = "axiong/imagenet-r"

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        train: bool = True,
    ):
        ds, classes, wnid_to_idx, all_targets, train_idx, test_idx = _imagenetr_split()
        self.classes = classes
        self._wnid_to_idx = wnid_to_idx

        idx = train_idx if train else test_idx
        self._ds = ds.select(idx)
        self.targets = [all_targets[i] for i in idx]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        row = self._ds[int(index)]
        image = row["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        target = self._wnid_to_idx[row["wnid"]]

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


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


@lru_cache(maxsize=None)
def _core50_targets(hf_split: str) -> list[int]:
    """Cache the (expensive to materialise) label column of a CORe50 Hub split."""
    return [int(x) for x in _load_hf(CORe50Dataset.HF_REPO, hf_split)["label"]]


@lru_cache(maxsize=None)
def _core50_valid_indices() -> list[int]:
    _, valid_idx = class_balanced_split(_core50_targets("train"), 0.1, seed=0)
    return [int(i) for i in valid_idx]


class CORe50Dataset(Dataset):
    """CORe50 hosted on the Hugging Face Hub (``adrake17/core50``).

    The Hub dataset provides ``train`` (131,892 images) and ``test`` (32,974
    images) splits over the 50 object classes. The ``valid`` split is carved from
    the Hub ``train`` split with a deterministic class-balanced 10% hold-out, and
    ``train&valid`` is the full Hub ``train`` split.
    """

    HF_REPO = "adrake17/core50"

    def __init__(
        self,
        root: str | Path | None = None,
        split: Literal["train", "train&valid", "valid", "test"] = "train",
        transform: Callable[..., Any] | None = None,
    ):
        if split == "test":
            self._ds = _load_hf(self.HF_REPO, "test")
            self.targets = _core50_targets("test")
        elif split in ("train", "train&valid"):
            self._ds = _load_hf(self.HF_REPO, "train")
            self.targets = _core50_targets("train")
        elif split == "valid":
            valid_idx = _core50_valid_indices()
            self._ds = _load_hf(self.HF_REPO, "train").select(valid_idx)
            train_targets = _core50_targets("train")
            self.targets = [train_targets[i] for i in valid_idx]
        else:
            raise ValueError(f"Unknown split: {split!r}")

        self.transform = transform
        self.classes = list(range(50))

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        row = self._ds[int(index)]
        image = row["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        target = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def valid_split_indices(
    n: int,
    validation_set: float,
    seed: int = 0,
) -> tuple[Sequence[int], Sequence[int]]:
    assert 0.0 < validation_set < 1.0
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).int()
    n_valid = int(n * validation_set)
    train_indices, valid_indices = indices[n_valid:], indices[:n_valid]
    logger.info(
        f"Splitting {n} samples into {len(train_indices)} train and {len(valid_indices)} valid"
    )
    return train_indices, valid_indices  # type: ignore


def class_balanced_split(
    targets: Sequence[int],
    split_size: float,
    seed: int = 0,
) -> tuple[Sequence[int], Sequence[int]]:
    assert 0.0 < split_size < 1.0
    targets = torch.tensor(targets).int()  # type: ignore
    classes = torch.unique(targets)
    train_indices = []
    valid_indices = []
    rng = torch.Generator().manual_seed(seed)
    for c in classes:
        class_indices = torch.where(targets == c)[0]
        class_indices = class_indices[torch.randperm(len(class_indices), generator=rng)]
        n_valid = int(len(class_indices) * split_size)
        valid_indices.append(class_indices[:n_valid])
        train_indices.append(class_indices[n_valid:])
    train_indices = torch.cat(train_indices)
    valid_indices = torch.cat(valid_indices)
    logger.info(
        f"Splitting {len(targets)} samples into {len(train_indices)} train and {len(valid_indices)} valid with class balance"
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
    if validation_set <= 0.0:
        train_dataset = ImageNetR(dataset_root, train_transform, train=True)
        test_dataset = ImageNetR(dataset_root, eval_transform, train=False)
    else:
        n = len(ImageNetR(dataset_root, train=True))
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


def SplitCORe50(
    dataset_root: str | Path = datasets_path(),
    n_experiences: int = 5,
    train_transform: Callable[..., Any] | None = None,
    eval_transform: Callable[..., Any] | None = None,
    seed: int | None = None,
    return_task_id: bool = False,
    shuffle: bool = True,
    validation_set: bool = False,
) -> CLScenario:
    if validation_set:
        train_dataset = CORe50Dataset(dataset_root, "train", train_transform)
        test_dataset = CORe50Dataset(dataset_root, "valid", eval_transform)
    else:
        train_dataset = CORe50Dataset(dataset_root, "train&valid", train_transform)
        test_dataset = CORe50Dataset(dataset_root, "test", eval_transform)

    # Default core50 test dataset is quite large so we downsample it
    test_n = 10_000
    rng = torch.Generator().manual_seed(0)
    test_indices = torch.randperm(len(test_dataset), generator=rng).int()[:test_n]
    test_dataset = Subset(test_dataset, test_indices)  # type: ignore

    return nc_benchmark(
        train_dataset=train_dataset,  # type: ignore
        test_dataset=test_dataset,  # type: ignore
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        shuffle=shuffle,
    )


def SplitCUB200_2011(
    dataset_root: str | Path = datasets_path(),
    n_experiences: int = 10,
    train_transform: Callable[..., Any] | None = None,
    eval_transform: Callable[..., Any] | None = None,
    seed: int | None = None,
    return_task_id: bool = False,
    shuffle: bool = True,
    validation_set: float = 0.0,
):
    root = Path(dataset_root) / "CUB_200_2011/CUB_200_2011/images"
    eval_set = 0.1
    train_dataset = ImageFolder(root, train_transform)
    valid_dataset = ImageFolder(root, eval_transform)
    test_dataset = ImageFolder(root, eval_transform)

    # Constant split for train/test set
    train_perm, test_perm = class_balanced_split(train_dataset.targets, eval_set, 1)  # type: ignore
    logger.info(
        f"Created (train & valid)/test split with {len(train_perm)} train and {len(test_perm)} test samples"
    )
    train_dataset = Subset(train_dataset, train_perm)
    valid_dataset = Subset(valid_dataset, train_perm)
    test_dataset = Subset(test_dataset, test_perm)

    if validation_set > 0.0:
        train_perm, valid_perm = valid_split_indices(
            len(train_dataset), validation_set, 2
        )
        logger.info(
            f"Created train/valid split with {len(train_perm)} train and {len(valid_perm)} valid samples"
        )
        train_dataset = Subset(train_dataset, train_perm)
        # REPLACE test dataset with valid dataset for evaluation during training
        test_dataset = Subset(valid_dataset, valid_perm)

    return nc_benchmark(
        train_dataset=train_dataset,  # type: ignore
        test_dataset=test_dataset,  # type: ignore
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        shuffle=shuffle,
    )
