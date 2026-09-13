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
from avalanche.benchmarks.scenarios.deprecated.generic_benchmark_creation import (
    create_generic_benchmark_from_paths,
)

from datasets import concatenate_datasets, load_dataset
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder


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
def _load_hf(repo: str, split: str, name: str | None = None):
    """Load and cache a Hugging Face dataset split for the lifetime of the process.

    ``Split*`` helpers instantiate each dataset class two or three times (a length
    probe plus the train and test views), so without this cache every call would
    re-open the Arrow table and re-run the class-balanced split. The cache is
    populated in the parent process before ``DataLoader`` workers fork, so workers
    inherit it for free; the underlying Arrow data is memory-mapped, not copied.

    ``name`` selects a dataset config (e.g. SVHN's ``cropped_digits`` vs.
    ``full_numbers``); most repos have only one and leave it ``None``.
    """
    return load_dataset(repo, name, split=split, cache_dir=hf_cache_dir())


@lru_cache(maxsize=None)
def _imagenetr_targets() -> tuple[list[str], dict[str, int], list[int]]:
    ds = _load_hf(ImageNetR.HF_REPO, "test")
    classes = sorted(set(ds["wnid"]))
    wnid_to_idx = {wnid: i for i, wnid in enumerate(classes)}
    all_targets = [wnid_to_idx[wnid] for wnid in ds["wnid"]]
    return classes, wnid_to_idx, all_targets


#: iImageNet-R200/10 split sizes: the Hub dataset's single 30,000-image pool,
#: class-balanced into train/valid/pilot_test/full_test.
IMAGENETR_SPLIT_SIZES = {
    "valid": 1250,
    "pilot_test": 1250,
    "full_test": 2500,
    "train": 25000,
}


@lru_cache(maxsize=None)
def _imagenetr_split_indices() -> dict[str, list[int]]:
    _, _, all_targets = _imagenetr_targets()
    return split_named(all_targets, IMAGENETR_SPLIT_SIZES, seed=0)


class ImageNetR(Dataset):
    """ImageNet-R(endition) hosted on the Hugging Face Hub (``axiong/imagenet-r``).

    The Hub dataset ships a single ``test`` split of 30,000 images labelled by
    WordNet id; integer labels are derived from the sorted list of WordNet ids.
    Exposes the full 30,000-image pool -- use :func:`split_named` /
    :data:`IMAGENETR_SPLIT_SIZES` (see :func:`SplitImageNetR`) to carve out
    train/valid/pilot_test/full_test partitions.
    """

    HF_REPO = "axiong/imagenet-r"

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
    ):
        classes, wnid_to_idx, all_targets = _imagenetr_targets()
        self.classes = classes
        self._wnid_to_idx = wnid_to_idx
        self._ds = _load_hf(self.HF_REPO, "test")
        self.targets = all_targets
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


def split_named(
    targets: Sequence[int],
    sizes: dict[str, int],
    seed: int = 0,
) -> dict[str, list[int]]:
    """Class-balanced partition of ``targets`` into named splits of the given sizes.

    Within each class, a share proportional to ``total/len(targets)`` is kept
    (dropping the rest if ``total < len(targets)``) and apportioned among the
    named splits by the largest-remainder method, so aggregate split sizes land
    within a sample or two of ``sizes`` even for skewed per-class counts --
    unlike sequential fractional peeling, whose truncation error compounds
    across classes and splits. ``sum(sizes.values())`` must not exceed
    ``len(targets)``.
    """
    total = sum(sizes.values())
    n = len(targets)
    assert total <= n, f"requested {total} samples from a pool of only {n}"
    names = list(sizes)
    fracs = [sizes[name] / total for name in names]

    targets_t = torch.tensor(targets).int()
    classes = torch.unique(targets_t)
    rng = torch.Generator().manual_seed(seed)
    out: dict[str, list[int]] = {name: [] for name in names}
    for c in classes:
        class_idx = torch.where(targets_t == c)[0]
        class_idx = class_idx[torch.randperm(len(class_idx), generator=rng)]
        keep = min(len(class_idx), round(len(class_idx) * total / n))
        kept = class_idx[:keep]

        raw = [keep * f for f in fracs]
        counts = [int(x) for x in raw]
        remainder = keep - sum(counts)
        # Give the leftover units to the splits with the largest fractional
        # part, so per-class allocations sum to ``keep`` exactly and aggregate
        # split sizes converge to ``sizes`` rather than always rounding down.
        order = sorted(range(len(names)), key=lambda i: raw[i] - counts[i], reverse=True)
        for i in order[:remainder]:
            counts[i] += 1

        cursor = 0
        for name, count in zip(names, counts):
            out[name].extend(int(i) for i in kept[cursor : cursor + count])
            cursor += count

    logger.info(
        f"Splitting {n} samples into {({k: len(v) for k, v in out.items()})} with class balance"
    )
    return out


def resolve_split_indices(
    named: dict[str, Sequence[int]],
    scale: Literal["pilot", "full"],
    validation: bool,
) -> tuple[list[int], list[int]]:
    """Resolve named train/valid/pilot_test/full_test partitions into a
    ``(train, eval)`` index pair for a given scale and stage.

    At ``full`` scale, the pilot's test set is recycled into the training pool
    since it is no longer needed for evaluation once the pilot has validated
    the setup. ``validation`` selects the held-out ``valid`` split (used for
    tuning/early stopping) instead of the scale's test split.
    """
    train = list(named["train"])
    if scale == "full":
        train = train + list(named["pilot_test"])
    if validation:
        eval_ = named["valid"]
    else:
        eval_ = named["full_test"] if scale == "full" else named["pilot_test"]
    return train, list(eval_)


def SplitImageNetR(
    dataset_root: str | Path = datasets_path(),
    n_experiences: int = 10,
    train_transform: Callable[..., Any] | None = None,
    eval_transform: Callable[..., Any] | None = None,
    seed: int | None = None,
    return_task_id: bool = False,
    shuffle: bool = True,
    scale: Literal["pilot", "full"] = "full",
    validation: bool = False,
) -> CLScenario:
    """Create iImageNet-R200/10 by splitting ImageNet-R(endition)[#f1].

    The Hub dataset's single 30,000-image pool is class-balanced into
    train/valid/pilot_test/full_test (:data:`IMAGENETR_SPLIT_SIZES`); ``scale``
    and ``validation`` select which partitions become the train/eval streams
    (see :func:`resolve_split_indices`).

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
    train_idx, eval_idx = resolve_split_indices(
        _imagenetr_split_indices(), scale, validation
    )
    train_dataset = Subset(ImageNetR(dataset_root, train_transform), train_idx)
    eval_dataset = Subset(ImageNetR(dataset_root, eval_transform), eval_idx)

    return nc_benchmark(
        train_dataset=train_dataset,  # type: ignore
        test_dataset=eval_dataset,  # type: ignore
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


@lru_cache(maxsize=None)
def _cifar100_pool():
    """The combined 60,000-image CIFAR-100 pool (Hub train + test splits).

    Re-partitioned into train/valid/pilot_test/full_test from scratch
    (40k/5k/5k/10k) rather than reusing the dataset's canonical 50k/10k
    train/test boundary, so pilot_test and full_test are a disjoint split of
    a dedicated "test" pool instead of full_test being the stock test set
    with pilot_test carved out of train.
    """
    return concatenate_datasets(
        [_load_hf(CIFAR100Dataset.HF_REPO, "train"), _load_hf(CIFAR100Dataset.HF_REPO, "test")]
    )


@lru_cache(maxsize=None)
def _cifar100_pool_targets() -> list[int]:
    return [int(x) for x in _cifar100_pool()["fine_label"]]


#: iCIFAR100/10 split sizes.
CIFAR100_SPLIT_SIZES = {"valid": 5000, "pilot_test": 5000, "full_test": 10000, "train": 40000}


@lru_cache(maxsize=None)
def _cifar100_split_indices() -> dict[str, list[int]]:
    return split_named(_cifar100_pool_targets(), CIFAR100_SPLIT_SIZES, seed=0)


class CIFAR100Dataset(Dataset):
    """CIFAR-100 hosted on the Hugging Face Hub (``uoft-cs/cifar100``).

    Exposes the full 60,000-image pool (Hub train + test splits combined) --
    use :func:`split_named` / :data:`CIFAR100_SPLIT_SIZES` (see
    :func:`SplitCIFAR100`) to carve out train/valid/pilot_test/full_test
    partitions.
    """

    HF_REPO = "uoft-cs/cifar100"

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable[..., Any] | None = None,
    ):
        self._ds = _cifar100_pool()
        self.targets = _cifar100_pool_targets()
        self.classes = list(range(100))
        self.transform = transform

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        row = self._ds[int(index)]
        image = row["img"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        target = int(row["fine_label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def SplitCIFAR100(
    dataset_root: str | Path = datasets_path(),
    n_experiences: int = 10,
    train_transform: Callable[..., Any] | None = None,
    eval_transform: Callable[..., Any] | None = None,
    seed: int | None = None,
    return_task_id: bool = False,
    shuffle: bool = True,
    scale: Literal["pilot", "full"] = "full",
    validation: bool = False,
) -> CLScenario:
    """Create iCIFAR100/10 by splitting CIFAR-100.

    The combined 60,000-image pool is class-balanced into train/valid/
    pilot_test/full_test (:data:`CIFAR100_SPLIT_SIZES`); ``scale`` and
    ``validation`` select which partitions become the train/eval streams
    (see :func:`resolve_split_indices`).
    """
    train_idx, eval_idx = resolve_split_indices(
        _cifar100_split_indices(), scale, validation
    )
    train_dataset = Subset(CIFAR100Dataset(dataset_root, train_transform), train_idx)
    eval_dataset = Subset(CIFAR100Dataset(dataset_root, eval_transform), eval_idx)

    return nc_benchmark(
        train_dataset=train_dataset,  # type: ignore
        test_dataset=eval_dataset,  # type: ignore
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        shuffle=shuffle,
    )


#: Per-bucket share of dCLEAR10/10's train/valid/pilot_test/full_test totals
#: (25000/1250/1250/2500 over 10 domains -- same totals as iImageNet-R200/10,
#: divided evenly across the 10 buckets). CLEAR10's raw buckets are far larger
#: than 3,000 images each, so :func:`split_named` also subsamples every bucket
#: down to this per-bucket total.
CLEAR10_SPLIT_SIZES = {"valid": 125, "pilot_test": 125, "full_test": 250, "train": 2500}


def _patch_clear10_download_urls() -> None:
    """Work around a URL bug in avalanche 0.6.0's ``clear_data.clear10`` entries.

    Each entry's ``base_url`` already includes the filename (e.g. ``.../
    clear10-train.zip``), but ``CLEARDataset._download_dataset`` does
    ``os.path.join(base_url, name)``, appending the filename a second time and
    404ing. The sibling ``clear10_neurips2021``/``clear100_cvpr2022`` entries
    don't have this bug since their ``base_url`` correctly ends in ``main/``.
    """
    from avalanche.benchmarks.datasets.clear import clear_data

    if clear_data.clear10[0][1].endswith(".zip"):
        clear_data.clear10 = [
            (name, url.rsplit("/", 1)[0] + "/") for name, url in clear_data.clear10
        ]


@lru_cache(maxsize=None)
def _clear10_buckets(
    dataset_root: str | Path = datasets_path(),
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Download CLEAR10 (Lin et al., 2021) and drop the 11th "BACKGROUND" class.

    The Hub release actually ships 11 time buckets (0-10), but bucket 0 only
    has metadata under ``test/`` -- ``train/`` starts at bucket 1 -- so it's a
    test-only reference bucket that was never meant to be trained on. Dropping
    it leaves exactly the 10 intended trainable domains.

    Returns 10 time buckets, each a tuple of ``(image_path, class_idx)`` with
    class indices remapped to the remaining 10 (non-background) classes.
    """
    _patch_clear10_download_urls()
    from avalanche.benchmarks.datasets.clear import _CLEARImage

    root = Path(dataset_root) / "clear10"
    ds = _CLEARImage(
        root=str(root), data_name="clear10", download=True, split="all", seed=None
    )

    background = next(
        i for i, name in enumerate(ds.class_names) if name.strip().upper() == "BACKGROUND"
    )
    remap = {
        old: new
        for new, old in enumerate(i for i in range(len(ds.class_names)) if i != background)
    }

    all_buckets = ds.get_paths_and_targets(root_appended=True)
    assert len(all_buckets) == 11, (
        f"expected CLEAR10's usual 11 raw time buckets (dropping the test-only "
        f"bucket 0 leaves the 10 intended domains), got {len(all_buckets)}"
    )

    return tuple(
        tuple(
            (str(path), remap[target])
            for path, target in bucket
            if target != background
        )
        for bucket in all_buckets[1:]
    )


@lru_cache(maxsize=None)
def _clear10_bucket_splits() -> tuple[dict[str, list[int]], ...]:
    buckets = _clear10_buckets()
    return tuple(
        split_named([target for _, target in bucket], CLEAR10_SPLIT_SIZES, seed=i)
        for i, bucket in enumerate(buckets)
    )


def SplitCLEAR10(
    dataset_root: str | Path = datasets_path(),
    train_transform: Callable[..., Any] | None = None,
    eval_transform: Callable[..., Any] | None = None,
    return_task_id: bool = False,
    scale: Literal["pilot", "full"] = "full",
    validation: bool = False,
) -> CLScenario:
    """Create dCLEAR10/10, a domain-incremental scenario over CLEAR10's 10 time
    buckets (Lin et al., 2021)[#f1], excluding the 11th "BACKGROUND" class.

    Unlike the class-incremental Split* benchmarks above, every bucket
    contains all 10 (non-background) classes, so experiences are the dataset's
    native time buckets rather than an ``nc_benchmark`` class partition. Each
    bucket is independently class-balanced into train/valid/pilot_test/
    full_test in the same proportions as iImageNet-R200/10; ``scale`` and
    ``validation`` select which partitions become the train/eval streams (see
    :func:`resolve_split_indices`).

    .. [#f1] Lin, Zhiqiu, et al. "The CLEAR Benchmark: Continual LEArning on
        Real-World Imagery." NeurIPS Datasets and Benchmarks. 2021.
    """
    buckets = _clear10_buckets(dataset_root)
    bucket_splits = _clear10_bucket_splits()

    train_paths: list[list[tuple[str, int]]] = []
    eval_paths: list[list[tuple[str, int]]] = []
    for bucket, named in zip(buckets, bucket_splits):
        train_idx, eval_idx = resolve_split_indices(named, scale, validation)
        train_paths.append([bucket[i] for i in train_idx])
        eval_paths.append([bucket[i] for i in eval_idx])

    task_labels = list(range(len(buckets))) if return_task_id else [0] * len(buckets)
    benchmark = create_generic_benchmark_from_paths(
        train_paths,
        eval_paths,
        task_labels=task_labels,
        complete_test_set_only=False,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )
    # Unlike NCScenario, the generic path-based scenario doesn't compute this
    # itself, but Experiment relies on ``benchmark.n_classes``.
    benchmark.n_classes = 1 + max(target for bucket in buckets for _, target in bucket)
    return benchmark


#: OOD dataset split sizes. There's no train/valid partition -- these
#: datasets are only ever used as auxiliary out-of-distribution eval sets,
#: never trained on.
OOD_SPLIT_SIZES = {"pilot_test": 5000, "full_test": 5000}


class CIFAR10Dataset(Dataset):
    """CIFAR-10 test split hosted on the Hugging Face Hub (``uoft-cs/cifar10``).

    Used as an auxiliary out-of-distribution dataset; see
    :func:`get_ood_dataset`.
    """

    HF_REPO = "uoft-cs/cifar10"

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable[..., Any] | None = None,
    ):
        self._ds = _load_hf(self.HF_REPO, "test")
        self.targets = [int(x) for x in self._ds["label"]]
        self.classes = list(range(10))
        self.transform = transform

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        row = self._ds[int(index)]
        image = row["img"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        target = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, target


class SVHNDataset(Dataset):
    """SVHN ``cropped_digits`` test split, hosted on the Hugging Face Hub
    (``ufldl-stanford/svhn``).

    Used as an auxiliary out-of-distribution dataset; see
    :func:`get_ood_dataset`.
    """

    HF_REPO = "ufldl-stanford/svhn"

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable[..., Any] | None = None,
    ):
        self._ds = _load_hf(self.HF_REPO, "test", name="cropped_digits")
        self.targets = [int(x) for x in self._ds["label"]]
        self.classes = list(range(10))
        self.transform = transform

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


_OOD_DATASETS: dict[str, type[Dataset]] = {
    "svhn": SVHNDataset,
    "cifar10": CIFAR10Dataset,
}


@lru_cache(maxsize=None)
def _ood_split_indices(name: str) -> dict[str, list[int]]:
    dataset_cls = _OOD_DATASETS[name]
    targets = dataset_cls().targets
    return split_named(targets, OOD_SPLIT_SIZES, seed=0)


def ood_dataset_names() -> list[str]:
    return list(_OOD_DATASETS)


def get_ood_dataset(
    name: str,
    scale: Literal["pilot", "full"],
    dataset_root: str | Path = datasets_path(),
    transform: Callable[..., Any] | None = None,
) -> Dataset:
    """Load an auxiliary out-of-distribution split.

    Used to compute ``auroc_$ood_dataset``: seen-task samples vs. this dataset,
    scored by max softmax probability. ``scale`` selects the disjoint
    pilot_test/full_test partition, matching the primary datasets' pilot/full
    protocol -- there's no recycling here since OOD sets are never trained on.
    """
    indices = _ood_split_indices(name)[f"{scale}_test"]
    return Subset(_OOD_DATASETS[name](dataset_root, transform), indices)


#: Corruption severities used for ``ece@$shift``/``ace@$shift``.
SHIFT_SEVERITIES = [1, 2, 3, 4, 5]


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
