"""All non-method configuration in one place.

Three layers, resolved into a single :class:`ExperimentConfig` per run:

* :class:`Backbone` / :class:`Dataset` (registry :data:`DATASETS`) -- the
  backbone, adapter filter, batch sizes, task count and per-dataset epoch
  budget. Replaces the old ``configs/base/dataset/*.jsonnet`` layer.
* :class:`Scale` (registry :data:`SCALES`) -- the tuning budget (``n_trials``),
  training length (``epochs``) and how many seeds ``test`` averages over.
* :class:`ExperimentConfig` -- the flat, typed config an
  :class:`~bayescl.experiment.Experiment` actually consumes, produced by
  :meth:`ExperimentConfig.from_spec` from a ``Dataset`` + ``Scale`` + CLI args.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Dataset definitions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Backbone:
    name: str = "microsoft/resnet-18"
    freeze_backbone: bool = True
    #: Regex selecting which submodules receive adapters.
    adapter_filter: str = r".*layer\.[0-9]\.convolution"
    head_module: str = "model.classifier.1"


@dataclass(frozen=True)
class Dataset:
    key: str
    #: Avalanche scenario name consumed by :func:`bayescl.benchmark.get_benchmark`.
    scenario: str
    n_tasks: int = 10
    shuffle: bool = False
    #: Training epochs per experience for the ``full`` scale.
    full_epochs: int = 30
    train_mb_size: int = 128
    eval_mb_size: int = 256
    num_workers: int = 4
    standardize: bool = True
    backbone: Backbone = field(default_factory=Backbone)


DATASETS: dict[str, Dataset] = {
    "cifar100": Dataset("cifar100", "CIFAR100", shuffle=True, full_epochs=30),
    # 224px decode+augment is CPU-bound; these benefit from more workers than the
    # default 4 (measured ~1.8x from 4->8 workers on both pipelines).
    "core50": Dataset("core50", "CORe50", full_epochs=30, num_workers=8),
    "imagenetr": Dataset(
        "imagenetr", "ImageNetR", shuffle=True, full_epochs=60, num_workers=8
    ),
    # CLEAR10's task order is chronological and meaningful, so it stays fixed.
    "clear10": Dataset("clear10", "CLEAR10", full_epochs=60, num_workers=8),
}


def dataset_names() -> list[str]:
    return list(DATASETS)


def get_dataset(key: str) -> Dataset:
    return DATASETS[key]


# --------------------------------------------------------------------------- #
# Experiment scale: ``pilot`` (fast smoke) vs ``full`` (paper-quality)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Scale:
    key: str
    #: Number of Optuna trials in the ``tune`` stage.
    n_trials: int
    #: Number of seeds the ``test`` stage runs and reports.
    n_seeds: int
    #: Epochs per experience. ``None`` -> use the dataset's ``full_epochs``.
    pilot_epochs: int | None

    def epochs(self, dataset: Dataset) -> int:
        return dataset.full_epochs if self.pilot_epochs is None else self.pilot_epochs


PILOT = Scale("pilot", n_trials=3, n_seeds=5, pilot_epochs=100)
# n_seeds is a floor: the pilot's variance estimates should drive a power
# analysis that may raise this count for a given dataset/metric combination.
FULL = Scale("full", n_trials=50, n_seeds=8, pilot_epochs=None)

SCALES: dict[str, Scale] = {s.key: s for s in (PILOT, FULL)}


def scale_names() -> list[str]:
    return list(SCALES)


def get_scale(key: str) -> Scale:
    return SCALES[key]


# --------------------------------------------------------------------------- #
# Resolved per-run config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    n_tasks: int
    shuffle: bool
    dataset_root: Path
    standardize: bool
    validation: bool
    scale: str
    backbone_name: str
    freeze_backbone: bool
    adapter_filter: str
    head_module: str
    epochs: int
    train_mb_size: int
    eval_mb_size: int
    num_workers: int
    run_dir: Path
    seed: int
    device: str

    #: Evaluate validation Brier every N epochs; stop once it hasn't improved for
    #: this many consecutive checks.
    early_stop_eval_every: int = 2
    early_stop_patience: int = 5

    @classmethod
    def from_spec(
        cls,
        dataset: Dataset,
        scale: Scale,
        *,
        seed: int,
        validation: bool,
        run_dir,
        dataset_root,
        device: str = "cuda",
    ) -> "ExperimentConfig":
        backbone = dataset.backbone
        return cls(
            dataset=dataset.scenario,
            n_tasks=dataset.n_tasks,
            shuffle=dataset.shuffle,
            dataset_root=Path(dataset_root),
            standardize=dataset.standardize,
            validation=validation,
            scale=scale.key,
            backbone_name=backbone.name,
            freeze_backbone=backbone.freeze_backbone,
            adapter_filter=backbone.adapter_filter,
            head_module=backbone.head_module,
            epochs=scale.epochs(dataset),
            train_mb_size=dataset.train_mb_size,
            eval_mb_size=dataset.eval_mb_size,
            num_workers=dataset.num_workers,
            run_dir=Path(run_dir),
            seed=seed,
            device=device,
        )
