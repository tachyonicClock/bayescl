import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List

import numpy as np
import torch
from sklearn.metrics import brier_score_loss
from torch import Tensor
from torchmetrics.utilities.compute import normalize_logits_if_needed

from .callibration import calibration_curve, expected_calibration_error

N_BINS = 15


def _backwards_transfer(R: torch.Tensor) -> float:
    n = R.size(0)
    assert R.shape == (n, n)
    return ((R - R.diag()).tril().sum() / (n * (n - 1) / 2)).item()


def _forwards_transfer(R: torch.Tensor) -> float:
    n = R.size(0)
    assert R.shape == (n, n)
    return (R.triu(1).sum() / (n * (n - 1) / 2)).item()


@dataclass(frozen=True)
class Result:
    r"""A collection of metrics evaluating an online continual learner.

    We define some metrics in terms of a matrix :math:`R\in\mathbb{R}^{T \times T}`
    (:attr:`accuracy_matrix`) where each element :math:`R_{i,j}` contains the
    the test accuracy on task :math:`j` after sequentially training on tasks
    :math:`1` through :math:`i`.
    """

    n_tasks: int
    r"""The number of tasks in the experiment."""
    accuracy_all: np.ndarray
    r"""The accuracy on all tasks after training on each task.
    
    .. math::

        a_\text{all}(t) = \frac{1}{T} \sum_{i=1}^{T} R_{t,i}

    Use :attr:`task_index` to get the corresponding task index for plotting.
    """
    accuracy_all_avg: float
    r"""The average of :attr:`accuracy_all` over all tasks.
    
    .. math::

        \bar{a}_\text{all} = \frac{1}{T}\sum_{t=1}^T a_\text{all}(t)
    """
    accuracy_seen: np.ndarray
    r"""The accuracy on **seen** tasks after training on each task.
    
    .. math::

        a_\text{seen}(t) = \frac{1}{t}\sum^t_{i=1} R_{t,i}

    Use :attr:`task_index` to get the corresponding task index for plotting.
    """
    accuracy_seen_avg: float
    r"""The average of :attr:`accuracy_seen` over all tasks.
    
    .. math::

        \bar{a}_\text{seen} = \frac{1}{T}\sum_{t=1}^T a_\text{seen}(t)
    """
    accuracy_final: float
    r"""The accuracy on all tasks after training on the final task.

    .. math::
    
        a_\text{final} = a_\text{all}(T)
    """
    task_index: np.ndarray
    r"""The position of each task in the metrics."""

    forward_transfer: float
    r"""A scalar measuring the impact learning had on future tasks.

    .. math::

       r_\text{FWT} = \frac{2}{T(T-1)}\sum_{i=1}^{T} \sum_{j=i+1}^{T} R_{i,j}
    """
    backward_transfer: float
    r"""A scalar measuring the impact learning had on past tasks.

    .. math::

       r_\text{BWT} = \frac{2}{T(T-1)} \sum_{i=2}^{T} \sum_{j=1}^{i-1} (R_{i,j} - R_{j,j})
    """
    accuracy_matrix: np.ndarray
    r"""A matrix measuring the accuracy on each task after training on each task.
    
    ``R[i, j]`` is the accuracy on task :math:`j` after training on tasks
    :math:`1` through :math:`i`.
    """

    def __post_init__(self):
        t = self.n_tasks
        assert self.accuracy_matrix.shape == (t, t)
        assert self.accuracy_all.shape == (t,)
        assert self.accuracy_seen.shape == (t,)
        assert self.task_index.shape == (t,)

    def save(self, f: BinaryIO | Path | str) -> None:
        if isinstance(f, (str, Path)):
            filename = Path(f)
            filename = filename.with_suffix(".pkl")
            with open(filename, "wb") as file:
                pickle.dump(asdict(self), file)
        elif isinstance(f, BinaryIO):
            pickle.dump(asdict(self), f)

    @classmethod
    @torch.no_grad()
    def from_accuracy_matrix(cls, R: Tensor | np.ndarray) -> "Result":
        """Create a Result object from task accuracy matrix.

        :param R: R matrix of shape (n_tasks, n_tasks) where each element
            :math:`R_{i,j}` contains the test accuracy on task :math:`j` after
            sequentially training on tasks :math:`1` through :math:`i`.
        :return: A Result object containing the metrics.
        """
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R)
        R = R.cpu()

        def _accuracy_seen(t: int) -> float:
            return R[t, : t + 1].mean().item()

        assert R.T.shape == R.shape, "R must be a square matrix."
        n_tasks = R.shape[0]
        accuracy_all = R.mean(dim=1)
        accuracy_all_avg = accuracy_all.mean().item()
        accuracy_seen = torch.tensor([_accuracy_seen(t) for t in range(0, n_tasks)])
        accuracy_seen_avg = accuracy_seen.mean().item()
        accuracy_final = accuracy_all[n_tasks - 1].item()
        task_index = torch.arange(n_tasks, dtype=torch.int64) + 1
        forward_transfer = _forwards_transfer(R)
        backward_transfer = _backwards_transfer(R)
        accuracy_matrix = R

        return cls(
            n_tasks=n_tasks,
            accuracy_all=accuracy_all.numpy(),
            accuracy_all_avg=accuracy_all_avg,
            accuracy_seen=accuracy_seen.numpy(),
            accuracy_seen_avg=accuracy_seen_avg,
            accuracy_final=accuracy_final,
            task_index=task_index.numpy(),
            forward_transfer=forward_transfer,
            backward_transfer=backward_transfer,
            accuracy_matrix=accuracy_matrix.numpy(),
        )


class ContinualLearningEvaluator:
    def __init__(
        self,
        num_tasks: int,
        num_classes: int,
    ) -> None:
        self._task_count = num_tasks
        self._class_count = num_classes
        self._big_r = torch.zeros(
            num_tasks,
            num_tasks,
            num_classes,
            num_classes,
            dtype=torch.long,
        )
        self._train_tid = 0
        self._test_tid = 0

        #: Dictionary mapping (train_task_idx, test_task_idx) to list of true labels
        self._y_true: Dict[tuple[int, int], List[Tensor]] = {}
        #: Dictionary mapping (train_task_idx, test_task_idx) to list of predicted logits
        self._y_logit: Dict[tuple[int, int], List[Tensor]] = {}

    @torch.no_grad()
    def update(
        self,
        train_task_idx: int,
        test_task_idx: int,
        y_logit: Tensor,
        y: Tensor,
    ):
        batch = y.shape[0]
        assert y_logit.shape == (batch, self._class_count)
        assert y.shape == (batch,)
        y_logit = y_logit.cpu()
        y = y.cpu()

        self._y_true.setdefault((train_task_idx, test_task_idx), []).append(y)
        self._y_logit.setdefault((train_task_idx, test_task_idx), []).append(y_logit)

        self._big_r[train_task_idx, test_task_idx].index_put_(
            (
                y.cpu().long(),
                y_logit.argmax(dim=1).cpu().long(),
            ),
            torch.tensor(1, dtype=torch.long),
            accumulate=True,
        )

    @staticmethod
    def ece(y_logit: Tensor, y_true: Tensor, num_bins: int = N_BINS) -> float:
        """Expected Calibration Error. Use probabilities from the predicted class only."""
        y_prob = normalize_logits_if_needed(y_logit, "softmax")
        bin_prob, bin_freq, bin_weights = calibration_curve(
            y_prob.numpy(),
            y_true.numpy(),
            num_bins=num_bins,
        )
        return expected_calibration_error(bin_prob, bin_freq, bin_weights)

    @staticmethod
    def sce(y_logit: Tensor, y_true: Tensor, num_bins: int = N_BINS) -> float:
        """Static Calibration Error. Use probabilities from all classes."""
        y_prob = normalize_logits_if_needed(y_logit, "softmax")
        bin_prob, bin_freq, bin_weights = calibration_curve(
            y_prob.numpy(),
            y_true.numpy(),
            num_bins=num_bins,
            top_class_only=False,  # gives results for the Static Calibration Error
        )
        return expected_calibration_error(bin_prob, bin_freq, bin_weights)

    @staticmethod
    def ace(y_logit: Tensor, y_true: Tensor, num_bins: int = N_BINS) -> float:
        """Adaptive Calibration Error. Bins have equal number of samples."""
        y_prob = normalize_logits_if_needed(y_logit, "softmax")
        bin_prob, bin_freq, bin_weights = calibration_curve(
            y_prob.numpy(),
            y_true.numpy(),
            num_bins=num_bins,
            top_class_only=False,
            equal_size_bins=True,  # gives results for the Adaptive Calibration Error
        )
        return expected_calibration_error(bin_prob, bin_freq, bin_weights)

    @staticmethod
    def brier(y_logit: Tensor, y_true: Tensor) -> float:
        """Brier score."""
        y_prob = normalize_logits_if_needed(y_logit, "softmax")
        return float(brier_score_loss(y_true.numpy(), y_prob.numpy()))

    @torch.no_grad()
    def result(self) -> Dict[str, Any]:
        y_true: Dict[tuple[int, int], Tensor] = {
            k: torch.cat(v, dim=0) for k, v in self._y_true.items()
        }
        y_logit: Dict[tuple[int, int], Tensor] = {
            k: torch.cat(v, dim=0) for k, v in self._y_logit.items()
        }

        # Squash and group by train task
        y_true_seen: List[Tensor] = []
        y_true_all: List[Tensor] = []
        y_logit_seen: List[Tensor] = []
        y_logit_all: List[Tensor] = []
        for train_tid in range(self._task_count):
            _y_true_seen = []
            _y_true_all = []
            _y_logit_seen = []
            _y_logit_all = []
            for test_tid in range(self._task_count):
                _y_true_all.append(y_true[(train_tid, test_tid)])
                _y_logit_all.append(y_logit[(train_tid, test_tid)])
                if train_tid >= test_tid:
                    _y_true_seen.append(y_true[(train_tid, test_tid)])
                    _y_logit_seen.append(y_logit[(train_tid, test_tid)])
            y_true_seen.append(torch.cat(_y_true_seen, dim=0))
            y_true_all.append(torch.cat(_y_true_all, dim=0))
            y_logit_seen.append(torch.cat(_y_logit_seen, dim=0))
            y_logit_all.append(torch.cat(_y_logit_all, dim=0))

        # Compute expected calibration errors per task
        ece_all = np.zeros(self._task_count)
        ece_seen = np.zeros(self._task_count)
        ace_all = np.zeros(self._task_count)
        ace_seen = np.zeros(self._task_count)
        sce_all = np.zeros(self._task_count)
        sce_seen = np.zeros(self._task_count)
        # brier_seen = np.zeros(self._task_count)
        brier_all = np.zeros(self._task_count)
        for t in range(self._task_count):
            ece_all[t] = self.ece(y_logit_all[t], y_true_all[t])
            ece_seen[t] = self.ece(y_logit_seen[t], y_true_seen[t])
            ace_all[t] = self.ace(y_logit_all[t], y_true_all[t])
            ace_seen[t] = self.ace(y_logit_seen[t], y_true_seen[t])
            sce_all[t] = self.sce(y_logit_all[t], y_true_all[t])
            sce_seen[t] = self.sce(y_logit_seen[t], y_true_seen[t])
            brier_all[t] = self.brier(y_logit_all[t], y_true_all[t])
            # brier_seen[t] = self.brier(y_logit_seen[t], y_true_seen[t])

        correct = self._big_r.diagonal(dim1=2, dim2=3).sum(dim=-1)
        total = self._big_r.sum(dim=(2, 3))
        accuracy = correct / total
        return {
            **asdict(Result.from_accuracy_matrix(accuracy)),
            # "y_true": {k: v.numpy() for k, v in y_true.items()},
            # "y_logit": {k: v.numpy() for k, v in y_logit.items()},
            "ece_all": ece_all,
            "ece_all_avg": ece_all.mean(),
            "ece_seen": ece_seen,
            "ece_seen_avg": ece_seen.mean(),
            "ece_final": ece_all[-1],
            "ace_all": ace_all,
            "ace_all_avg": ace_all.mean(),
            "ace_seen": ace_seen,
            "ace_seen_avg": ace_seen.mean(),
            "ace_final": ace_all[-1],
            "sce_all": sce_all,
            "sce_all_avg": sce_all.mean(),
            "sce_seen": sce_seen,
            "sce_seen_avg": sce_seen.mean(),
            "sce_final": sce_all[-1],
            "brier_all": brier_all,
            "brier_all_avg": brier_all.mean(),
            "brier_final": brier_all[-1],
        }
