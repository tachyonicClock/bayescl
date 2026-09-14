"""Pairwise significance testing between treatments, and pilot-driven
sample-size determination for the ``full``-scale ``test`` stage.

Comparisons are two-sided Wilcoxon signed-rank tests between each of our
methods and each baseline (plus each pair among our methods), corrected for
multiple comparisons with the Holm-Bonferroni step-down procedure. As a
paired test, each comparison matches observations by seed index: the ``test``
stage always runs seeds ``0..n_seeds-1`` for every method/dataset at a given
scale, and task order is a function of the seed and dataset alone (not the
method), so ``seed_NN`` for method A and ``seed_NN`` for method B share the
same initialization seed and task ordering.
"""

from __future__ import annotations

import itertools
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np
from scipy.stats import norm, wilcoxon

from bayescl.runio import latest_run

OUR_METHODS = ("ball", "tball", "tball_mnd")
BASELINES = ("lora", "clora", "ewc", "inflora", "rwalk", "sdlora")
DATASETS = ("cifar100", "imagenetr", "clear10")
OOD_DATASETS = ("svhn", "cifar10")
SHIFT_SEVERITIES = (1, 2, 3, 4, 5)

MIN_TEST_RUNS = 8


def _brier(metrics: dict) -> float:
    return metrics["brier_seen_avg"]


def _mean_ood_auroc(metrics: dict) -> float:
    return float(np.mean([metrics[f"auroc_{name}_avg"] for name in OOD_DATASETS]))


def _mean_shift_ece(metrics: dict) -> float:
    return float(np.mean([metrics[f"ece_shift_{s}_avg"] for s in SHIFT_SEVERITIES]))


#: The three endpoints compared in the analysis: the tuning objective, mean
#: OOD-detection AUROC, and mean calibration error under distribution shift.
ENDPOINTS: Dict[str, Callable[[dict], float]] = {
    "brier": _brier,
    "auroc_ood": _mean_ood_auroc,
    "ece_shift": _mean_shift_ece,
}


def our_vs_others_pairs() -> list[tuple[str, str]]:
    """Every pair among our methods, plus every (ours, baseline) pair."""
    return [
        *itertools.combinations(OUR_METHODS, 2),
        *itertools.product(OUR_METHODS, BASELINES),
    ]


#: Total comparisons in the family-wise correction: pairs x datasets x endpoints.
N_COMPARISONS = len(our_vs_others_pairs()) * len(DATASETS) * len(ENDPOINTS)


def load_seed_values(test_run_dir: Path, endpoint: str) -> np.ndarray:
    """Per-seed endpoint values from a ``test`` run directory (one that
    contains ``seed_NN/metrics.pkl`` subdirectories, as written by
    ``Experiment.run``)."""
    fn = ENDPOINTS[endpoint]
    values = []
    for seed_dir in sorted(test_run_dir.glob("seed_*")):
        with open(seed_dir / "metrics.pkl", "rb") as f:
            values.append(fn(pickle.load(f)))
    return np.array(values)


def load_method_values(
    runs_dir: Path, scale: str, dataset: str, method: str, endpoint: str
) -> np.ndarray:
    """Per-seed endpoint values from the most recent ``test`` run for a given
    scale/dataset/method."""
    run_dir = latest_run(runs_dir / "test" / scale / dataset / method)
    return load_seed_values(run_dir, endpoint)


@dataclass(frozen=True)
class Comparison:
    dataset: str
    endpoint: str
    method_a: str
    method_b: str
    statistic: float
    p_value: float
    significant: bool = False


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni step-down correction.

    Returns a boolean array (aligned with ``p_values``) marking which
    hypotheses are rejected (i.e. statistically significant) at family-wise
    level ``alpha``.
    """
    p_values_ = np.asarray(p_values, dtype=float)
    m = len(p_values_)
    order = np.argsort(p_values_)
    sorted_p = p_values_[order]
    thresholds = alpha / (m - np.arange(m))
    passes = sorted_p <= thresholds
    if not passes.all():
        first_failure = int(np.argmax(~passes))
        passes[first_failure:] = False
    rejected = np.zeros(m, dtype=bool)
    rejected[order] = passes
    return rejected


def run_comparisons(runs_dir: Path, scale: str = "full") -> list[Comparison]:
    """Paired Wilcoxon signed-rank comparisons for every (dataset, endpoint,
    method pair), Holm-Bonferroni corrected across the full family of
    comparisons. Pairing relies on both methods' ``test`` runs having used
    the same seed sequence (see module docstring)."""
    comparisons = []
    for dataset in DATASETS:
        for endpoint in ENDPOINTS:
            values = {
                method: load_method_values(runs_dir, scale, dataset, method, endpoint)
                for method in (*OUR_METHODS, *BASELINES)
            }
            for method_a, method_b in our_vs_others_pairs():
                statistic, p_value = wilcoxon(
                    values[method_a], values[method_b], alternative="two-sided"
                )
                comparisons.append(
                    Comparison(
                        dataset, endpoint, method_a, method_b, statistic, p_value
                    )
                )

    rejected = holm_bonferroni([c.p_value for c in comparisons])
    return [
        Comparison(
            c.dataset,
            c.endpoint,
            c.method_a,
            c.method_b,
            c.statistic,
            c.p_value,
            bool(sig),
        )
        for c, sig in zip(comparisons, rejected)
    ]


def required_n_per_group(
    delta: float, sigma: float, alpha: float, power: float = 0.8
) -> int:
    """Minimum number of matched (paired) runs for a two-sided Wilcoxon
    signed-rank test to detect a shift of ``delta`` given the standard
    deviation ``sigma`` of the paired differences, using Noether's (1987)
    normal-approximation formula for the signed-rank test.
    """
    if sigma <= 0:
        return 0
    p = norm.cdf(delta * np.sqrt(2) / sigma)
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    n = ((z_alpha + z_beta) ** 2) / (12 * (p - 0.5) ** 2)
    return int(np.ceil(n))


def required_test_runs(runs_dir: Path, delta: float = 0.02, power: float = 0.8) -> int:
    """Number of ``test``-stage seeds the ``full`` scale should run, sized
    from the pilot scale's variance of paired differences so that a
    ``delta``-sized absolute difference in any endpoint is detectable at
    ``power`` under the strictest Holm-corrected significance level, with a
    floor of :data:`MIN_TEST_RUNS`.
    """
    alpha = 0.05 / N_COMPARISONS
    required = MIN_TEST_RUNS
    for dataset in DATASETS:
        for endpoint in ENDPOINTS:
            values = {
                method: load_method_values(runs_dir, "pilot", dataset, method, endpoint)
                for method in (*OUR_METHODS, *BASELINES)
            }
            for method_a, method_b in our_vs_others_pairs():
                a, b = values[method_a], values[method_b]
                sigma = float((a - b).std(ddof=1))
                required = max(
                    required, required_n_per_group(delta, sigma, alpha, power)
                )
    return required
