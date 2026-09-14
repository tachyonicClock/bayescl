import pickle
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss, roc_auc_score
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
    ece_all: np.ndarray | None = None
    r"""Expected calibration error over all tasks at each checkpoint."""
    ece_all_avg: float | None = None
    r"""Mean of :attr:`ece_all` across checkpoints."""
    ece_seen: np.ndarray | None = None
    r"""Expected calibration error averaged over seen tasks at each checkpoint."""
    ece_seen_avg: float | None = None
    r"""Mean of :attr:`ece_seen` across checkpoints."""
    ece_final: float | None = None
    r"""Expected calibration error over all tasks at the final checkpoint."""
    ace_all: np.ndarray | None = None
    r"""Adaptive calibration error over all tasks at each checkpoint."""
    ace_all_avg: float | None = None
    r"""Mean of :attr:`ace_all` across checkpoints."""
    ace_seen: np.ndarray | None = None
    r"""Adaptive calibration error averaged over seen tasks at each checkpoint."""
    ace_seen_avg: float | None = None
    r"""Mean of :attr:`ace_seen` across checkpoints."""
    ace_final: float | None = None
    r"""Adaptive calibration error over all tasks at the final checkpoint."""
    sce_all: np.ndarray | None = None
    r"""Static calibration error over all tasks at each checkpoint."""
    sce_all_avg: float | None = None
    r"""Mean of :attr:`sce_all` across checkpoints."""
    sce_seen: np.ndarray | None = None
    r"""Static calibration error averaged over seen tasks at each checkpoint."""
    sce_seen_avg: float | None = None
    r"""Mean of :attr:`sce_seen` across checkpoints."""
    sce_final: float | None = None
    r"""Static calibration error over all tasks at the final checkpoint."""
    brier_all: np.ndarray | None = None
    r"""Brier score over all tasks at each checkpoint."""
    brier_all_avg: float | None = None
    r"""Mean of :attr:`brier_all` across checkpoints."""
    brier_seen: np.ndarray | None = None
    r"""Brier score averaged over seen tasks at each checkpoint."""
    brier_seen_avg: float | None = None
    r"""Mean of :attr:`brier_seen` across checkpoints."""
    brier_final: float | None = None
    r"""Brier score over all tasks at the final checkpoint."""
    nll_all: np.ndarray | None = None
    r"""Negative log-likelihood over all tasks at each checkpoint."""
    nll_all_avg: float | None = None
    r"""Mean of :attr:`nll_all` across checkpoints."""
    nll_seen: np.ndarray | None = None
    r"""Negative log-likelihood averaged over seen tasks at each checkpoint."""
    nll_seen_avg: float | None = None
    r"""Mean of :attr:`nll_seen` across checkpoints."""
    nll_final: float | None = None
    r"""Negative log-likelihood over all tasks at the final checkpoint."""
    auroc_future: np.ndarray | None = None
    r"""AUROC for seen versus future-task samples at each non-final checkpoint."""
    auroc_future_avg: float | None = None
    r"""Mean of :attr:`auroc_future` across non-final checkpoints."""
    auroc_ood: Dict[str, np.ndarray] | None = None
    r"""AUROC arrays for seen-task versus auxiliary OOD datasets, keyed by name."""
    auroc_ood_avg: Dict[str, float] | None = None
    r"""Mean OOD AUROC values keyed by auxiliary dataset name."""
    mean_auroc_ood: float | None = None
    r"""Mean AUROC across the recorded auxiliary OOD datasets."""
    ece_shift: Dict[int, np.ndarray] | None = None
    r"""Shifted-data ECE arrays keyed by corruption severity."""
    ece_shift_avg: Dict[int, float] | None = None
    r"""Mean shifted-data ECE values keyed by corruption severity."""
    mean_ece_shift: float | None = None
    r"""Mean shifted-data ECE across the recorded corruption severities."""
    ace_shift: Dict[int, np.ndarray] | None = None
    r"""Shifted-data ACE arrays keyed by corruption severity."""
    ace_shift_avg: Dict[int, float] | None = None
    r"""Mean shifted-data ACE values keyed by corruption severity."""
    duration_s: float | None = None
    r"""Elapsed time used to compute the evaluation metrics."""

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
        #: Dictionary mapping (severity, train_task_idx) to logits/labels on
        #: seen-task samples replaced by a shifted version (``ece@$shift``).
        self._shift_logit: Dict[tuple[int, int], List[Tensor]] = {}
        self._shift_true: Dict[tuple[int, int], List[Tensor]] = {}
        #: Dictionary mapping (ood_dataset_name, train_task_idx) to logits on
        #: an auxiliary OOD dataset (``auroc_$ood_dataset``).
        self._ood_logit: Dict[tuple[str, int], List[Tensor]] = {}
        self._start_time = time.perf_counter()

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

    @torch.no_grad()
    def record_shift(
        self, severity: int, train_task_idx: int, y_logit: Tensor, y_true: Tensor
    ) -> None:
        """Record predictions on seen-task samples replaced by a version
        corrupted at ``severity``, for the ``ece@$shift``/``ace@$shift`` metrics.
        """
        key = (severity, train_task_idx)
        self._shift_logit.setdefault(key, []).append(y_logit.cpu())
        self._shift_true.setdefault(key, []).append(y_true.cpu())

    @torch.no_grad()
    def record_ood(self, name: str, train_task_idx: int, y_logit: Tensor) -> None:
        """Record predictions on an auxiliary out-of-distribution dataset,
        for the ``auroc_$ood_dataset`` metric."""
        key = (name, train_task_idx)
        self._ood_logit.setdefault(key, []).append(y_logit.cpu())

    @torch.no_grad()
    def intermediate_result(self, t: int) -> float:
        """Compute the seen-task Brier score after training on task ``t``."""
        brier_sum = 0.0
        for t_prime in range(t + 1):
            per_task_brier = [
                self.brier(
                    torch.cat(self._y_logit[(t_prime, j)], dim=0),
                    torch.cat(self._y_true[(t_prime, j)], dim=0),
                )
                for j in range(t_prime + 1)
                if (t_prime, j) in self._y_true
            ]
            brier_sum += np.mean(per_task_brier) if per_task_brier else 0.0

        return brier_sum / (t + 1)

    @torch.no_grad()
    def checkpoint_brier_scores(self, train_task_idx: int) -> tuple[float, float]:
        """Return the all-task and seen-task Brier scores at a checkpoint."""
        scores = [
            self.brier(
                torch.cat(self._y_logit[(train_task_idx, test_task_idx)], dim=0),
                torch.cat(self._y_true[(train_task_idx, test_task_idx)], dim=0),
            )
            for test_task_idx in range(self._task_count)
            if (train_task_idx, test_task_idx) in self._y_true
        ]
        seen_scores = [
            self.brier(
                torch.cat(self._y_logit[(train_task_idx, test_task_idx)], dim=0),
                torch.cat(self._y_true[(train_task_idx, test_task_idx)], dim=0),
            )
            for test_task_idx in range(train_task_idx + 1)
            if (train_task_idx, test_task_idx) in self._y_true
        ]
        return float(np.mean(scores)), float(np.mean(seen_scores))

    @torch.no_grad()
    def per_task_scores(self, train_task_idx: int) -> Dict[int, Dict[str, float]]:
        """Brier and ECE for each test task evaluated at this checkpoint,
        from the logits already captured by the per-checkpoint eval on
        ``self.benchmark.test_stream`` -- no extra eval pass needed."""
        scores = {}
        for test_task_idx in range(train_task_idx + 1):
            key = (train_task_idx, test_task_idx)
            if key not in self._y_true:
                continue
            logit = torch.cat(self._y_logit[key], dim=0)
            true = torch.cat(self._y_true[key], dim=0)
            scores[test_task_idx] = {
                "brier": self.brier(logit, true),
                "ece": self.ece(logit, true),
            }
        return scores

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
        """Brier score.

        Passes ``labels`` explicitly: sklearn otherwise infers the class set
        from ``y_true`` alone, which breaks as soon as a batch doesn't
        contain every class -- always true for ``brier_seen`` early in
        training, when only a handful of tasks (hence classes) have been seen.
        """
        y_prob = normalize_logits_if_needed(y_logit, "softmax")
        labels = list(range(y_logit.shape[1]))
        return float(brier_score_loss(y_true.numpy(), y_prob.numpy(), labels=labels))

    @staticmethod
    def nll(y_logit: Tensor, y_true: Tensor) -> float:
        """Mean negative log-likelihood."""
        log_prob = F.log_softmax(y_logit, dim=1)
        return float(F.nll_loss(log_prob, y_true))

    @staticmethod
    def auroc(id_logit: Tensor, ood_logit: Tensor) -> float:
        """AUROC distinguishing in-distribution (``id_logit``) samples from
        out-of-distribution (``ood_logit``) samples, scored by max softmax
        probability with in-distribution as the positive class."""
        id_score = normalize_logits_if_needed(id_logit, "softmax").amax(dim=1)
        ood_score = normalize_logits_if_needed(ood_logit, "softmax").amax(dim=1)
        score = torch.cat([id_score, ood_score]).numpy()
        label = np.concatenate([np.ones(id_score.shape[0]), np.zeros(ood_score.shape[0])])
        return float(roc_auc_score(label, score))

    @torch.no_grad()
    def result(self) -> Result:
        y_true: Dict[tuple[int, int], Tensor] = {
            k: torch.cat(v, dim=0) for k, v in self._y_true.items()
        }
        y_logit: Dict[tuple[int, int], Tensor] = {
            k: torch.cat(v, dim=0) for k, v in self._y_logit.items()
        }

        # Per-(train, test) task metric matrices, mirroring the accuracy
        # matrix R: entry [train_tid, test_tid] is the metric computed only
        # from that one test task's samples. "_all"/"_seen" then average
        # across test tasks the same way `accuracy_all`/`accuracy_seen` do,
        # so early (resp. late) tasks aren't overweighted just because a
        # checkpoint's seen (resp. future) task pool happens to hold more
        # raw samples than the others.
        T = self._task_count
        ece_matrix = np.zeros((T, T))
        ace_matrix = np.zeros((T, T))
        sce_matrix = np.zeros((T, T))
        brier_matrix = np.zeros((T, T))
        nll_matrix = np.zeros((T, T))
        for train_tid in range(T):
            for test_tid in range(T):
                logit = y_logit[(train_tid, test_tid)]
                true = y_true[(train_tid, test_tid)]
                ece_matrix[train_tid, test_tid] = self.ece(logit, true)
                ace_matrix[train_tid, test_tid] = self.ace(logit, true)
                sce_matrix[train_tid, test_tid] = self.sce(logit, true)
                brier_matrix[train_tid, test_tid] = self.brier(logit, true)
                nll_matrix[train_tid, test_tid] = self.nll(logit, true)

        def _all(matrix: np.ndarray) -> np.ndarray:
            return matrix.mean(axis=1)

        def _seen(matrix: np.ndarray) -> np.ndarray:
            return np.array([matrix[t, : t + 1].mean() for t in range(T)])

        ece_all, ece_seen = _all(ece_matrix), _seen(ece_matrix)
        ace_all, ace_seen = _all(ace_matrix), _seen(ace_matrix)
        sce_all, sce_seen = _all(sce_matrix), _seen(sce_matrix)
        brier_all, brier_seen = _all(brier_matrix), _seen(brier_matrix)
        nll_all, nll_seen = _all(nll_matrix), _seen(nll_matrix)

        # Pooled (not per-task-averaged) seen-task logits: AUROC is a
        # sample-level ranking metric, so the ID group is naturally the raw
        # union of every seen task's samples rather than an average of
        # per-task AUROC values.
        y_logit_seen: List[Tensor] = [
            torch.cat([y_logit[(t, j)] for j in range(t + 1)], dim=0) for t in range(T)
        ]

        correct = self._big_r.diagonal(dim1=2, dim2=3).sum(dim=-1)
        total = self._big_r.sum(dim=(2, 3))
        accuracy = correct / total
        accuracy_all = accuracy.mean(dim=1)
        accuracy_seen = torch.tensor(
            [accuracy[t, : t + 1].mean() for t in range(T)]
        )
        result = Result(
            n_tasks=T,
            accuracy_all=accuracy_all.numpy(),
            accuracy_all_avg=accuracy_all.mean().item(),
            accuracy_seen=accuracy_seen.numpy(),
            accuracy_seen_avg=accuracy_seen.mean().item(),
            accuracy_final=accuracy_all[-1].item(),
            task_index=np.arange(1, T + 1),
            forward_transfer=_forwards_transfer(accuracy),
            backward_transfer=_backwards_transfer(accuracy),
            accuracy_matrix=accuracy.numpy(),
        )
        metrics = {
            **asdict(result),
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
            "brier_seen": brier_seen,
            "brier_seen_avg": brier_seen.mean(),
            "brier_final": brier_all[-1],
            "nll_all": nll_all,
            "nll_all_avg": nll_all.mean(),
            "nll_seen": nll_seen,
            "nll_seen_avg": nll_seen.mean(),
            "nll_final": nll_all[-1],
            "duration_s": time.perf_counter() - self._start_time,
        }

        # auroc_future: seen tasks vs. pooled future tasks at each checkpoint,
        # excluding the final checkpoint (which has no future tasks left).
        if T > 1:
            auroc_future = np.array(
                [
                    self.auroc(
                        torch.cat([y_logit[(t, j)] for j in range(t + 1)], dim=0),
                        torch.cat([y_logit[(t, j)] for j in range(t + 1, T)], dim=0),
                    )
                    for t in range(T - 1)
                ]
            )
            metrics["auroc_future"] = auroc_future
            metrics["auroc_future_avg"] = auroc_future.mean()

        # auroc_$ood_dataset: seen tasks vs. an auxiliary OOD dataset, at every
        # checkpoint it was recorded at (see MetricsPlugin.capture / Experiment.run).
        ood_names = sorted({name for name, _ in self._ood_logit})
        for name in ood_names:
            values = [
                self.auroc(y_logit_seen[t], torch.cat(self._ood_logit[(name, t)], dim=0))
                for t in range(T)
                if (name, t) in self._ood_logit
            ]
            if values:
                metrics[f"auroc_{name}"] = np.array(values)
                metrics[f"auroc_{name}_avg"] = float(np.mean(values))

        # ece@$shift / ace@$shift: seen-task samples replaced by a corrupted
        # version at a given severity, at every checkpoint it was recorded at.
        severities = sorted({severity for severity, _ in self._shift_logit})
        for severity in severities:
            ece_values, ace_values = [], []
            for t in range(T):
                key = (severity, t)
                if key not in self._shift_logit:
                    continue
                shift_logit = torch.cat(self._shift_logit[key], dim=0)
                shift_true = torch.cat(self._shift_true[key], dim=0)
                ece_values.append(self.ece(shift_logit, shift_true))
                ace_values.append(self.ace(shift_logit, shift_true))
            if ece_values:
                metrics[f"ece_shift_{severity}"] = np.array(ece_values)
                metrics[f"ece_shift_{severity}_avg"] = float(np.mean(ece_values))
                metrics[f"ace_shift_{severity}"] = np.array(ace_values)
                metrics[f"ace_shift_{severity}_avg"] = float(np.mean(ace_values))

        result = replace(
            result,
            **{
                key: metrics[key]
                for key in (
                    "ece_all", "ece_all_avg", "ece_seen", "ece_seen_avg", "ece_final",
                    "ace_all", "ace_all_avg", "ace_seen", "ace_seen_avg", "ace_final",
                    "sce_all", "sce_all_avg", "sce_seen", "sce_seen_avg", "sce_final",
                    "brier_all", "brier_all_avg", "brier_seen", "brier_seen_avg", "brier_final",
                    "nll_all", "nll_all_avg", "nll_seen", "nll_seen_avg", "nll_final",
                    "duration_s",
                )
            },
            auroc_future=metrics.get("auroc_future"),
            auroc_future_avg=metrics.get("auroc_future_avg"),
            auroc_ood={
                name: metrics[f"auroc_{name}"] for name in ood_names
            } or None,
            auroc_ood_avg={
                name: metrics[f"auroc_{name}_avg"] for name in ood_names
            } or None,
            mean_auroc_ood=(
                float(np.mean([metrics[f"auroc_{name}_avg"] for name in ood_names]))
                if any(f"auroc_{name}_avg" in metrics for name in ood_names)
                else None
            ),
            ece_shift={
                severity: metrics[f"ece_shift_{severity}"] for severity in severities
                if f"ece_shift_{severity}" in metrics
            } or None,
            ece_shift_avg={
                severity: metrics[f"ece_shift_{severity}_avg"] for severity in severities
                if f"ece_shift_{severity}_avg" in metrics
            } or None,
            mean_ece_shift=(
                float(
                    np.mean(
                        [
                            metrics[f"ece_shift_{severity}_avg"]
                            for severity in severities
                            if f"ece_shift_{severity}_avg" in metrics
                        ]
                    )
                )
                if any(f"ece_shift_{severity}_avg" in metrics for severity in severities)
                else None
            ),
            ace_shift={
                severity: metrics[f"ace_shift_{severity}"] for severity in severities
                if f"ace_shift_{severity}" in metrics
            } or None,
            ace_shift_avg={
                severity: metrics[f"ace_shift_{severity}_avg"] for severity in severities
                if f"ace_shift_{severity}_avg" in metrics
            } or None,
        )

        return result
