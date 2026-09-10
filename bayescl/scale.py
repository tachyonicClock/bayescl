"""Experiment scale: ``pilot`` (fast smoke) vs ``full`` (paper-quality).

The scale controls the tuning budget (``n_trials``), the training length
(``epochs``) and how many seeds the ``test`` stage averages over (``n_seeds``).
"""

from __future__ import annotations

from dataclasses import dataclass

from bayescl.datasets_spec import Dataset


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


PILOT = Scale("pilot", n_trials=4, n_seeds=1, pilot_epochs=2)
FULL = Scale("full", n_trials=30, n_seeds=3, pilot_epochs=None)

SCALES: dict[str, Scale] = {s.key: s for s in (PILOT, FULL)}


def scale_names() -> list[str]:
    return list(SCALES)


def get_scale(key: str) -> Scale:
    return SCALES[key]
