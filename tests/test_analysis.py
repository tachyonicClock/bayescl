import pickle

import numpy as np
import pytest

from bayescl.analysis import (
    BASELINES,
    DATASETS,
    ENDPOINTS,
    N_COMPARISONS,
    OOD_DATASETS,
    OUR_METHODS,
    SHIFT_SEVERITIES,
    holm_bonferroni,
    load_method_values,
    our_vs_others_pairs,
    required_n_per_group,
    required_test_runs,
    run_comparisons,
)


def test_holm_bonferroni_matches_hand_computed_example():
    # Textbook example: 4 hypotheses, sorted p-values [0.01, 0.02, 0.03, 0.04].
    # Thresholds are alpha/4, alpha/3, alpha/2, alpha/1 = [.0125, .0167, .025, .05].
    # 0.01 <= .0125 (reject), 0.02 > .0167 (fail -> stop, nothing after rejected).
    p_values = [0.03, 0.01, 0.04, 0.02]
    rejected = holm_bonferroni(p_values, alpha=0.05)
    assert rejected.tolist() == [False, True, False, False]


def test_holm_bonferroni_rejects_all_when_all_tiny():
    rejected = holm_bonferroni([1e-9, 1e-8, 1e-7], alpha=0.05)
    assert rejected.all()


def test_required_n_per_group_decreases_with_larger_delta():
    small_delta_n = required_n_per_group(delta=0.01, sigma=0.1, alpha=0.05)
    large_delta_n = required_n_per_group(delta=0.05, sigma=0.1, alpha=0.05)
    assert large_delta_n < small_delta_n


def test_required_n_per_group_increases_with_smaller_alpha():
    loose_n = required_n_per_group(delta=0.02, sigma=0.1, alpha=0.05)
    strict_n = required_n_per_group(delta=0.02, sigma=0.1, alpha=0.05 / 189)
    assert strict_n > loose_n


def test_required_n_per_group_zero_sigma_is_zero():
    assert required_n_per_group(delta=0.02, sigma=0.0, alpha=0.05) == 0


def test_our_vs_others_pairs_matches_spec_comparison_count():
    n_our_methods = len(OUR_METHODS)
    n_baselines = len(BASELINES)
    expected_pairs = (n_our_methods * (n_our_methods - 1)) // 2 + n_our_methods * n_baselines
    assert len(our_vs_others_pairs()) == expected_pairs
    assert N_COMPARISONS == expected_pairs * len(DATASETS) * len(ENDPOINTS)


def _write_seed_metrics(run_dir, seed: int, brier: float, auroc: float, ece_shift: float) -> None:
    seed_dir = run_dir / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True)
    metrics = {
        "brier_seen_avg": brier,
        **{f"auroc_{name}_avg": auroc for name in OOD_DATASETS},
        **{f"ece_shift_{s}_avg": ece_shift for s in SHIFT_SEVERITIES},
    }
    with open(seed_dir / "metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)


def _populate_runs(tmp_path, scale: str, n_seeds: int, method_means: dict) -> None:
    rng = np.random.default_rng(0)
    for dataset in DATASETS:
        for method, (brier_mean, auroc_mean, shift_mean) in method_means.items():
            run_dir = tmp_path / "test" / scale / dataset / method / "20260101_000000"
            for seed in range(n_seeds):
                _write_seed_metrics(
                    run_dir,
                    seed,
                    brier_mean + rng.normal(scale=0.01),
                    auroc_mean + rng.normal(scale=0.01),
                    shift_mean + rng.normal(scale=0.01),
                )


def _all_methods_close_together() -> dict:
    return {method: (0.5, 0.7, 0.3) for method in (*OUR_METHODS, *BASELINES)}


def test_load_method_values_reads_pilot_seed_metrics(tmp_path):
    method_means = _all_methods_close_together()
    _populate_runs(tmp_path, "pilot", n_seeds=5, method_means=method_means)

    values = load_method_values(tmp_path, "pilot", DATASETS[0], "ball", "brier")
    assert values.shape == (5,)
    assert values == pytest.approx(0.5, abs=0.05)


def test_run_comparisons_flags_a_clearly_separated_pair(tmp_path):
    method_means = _all_methods_close_together()
    # Give "ball" a clearly better (lower) brier than every baseline on every dataset.
    method_means["ball"] = (0.1, 0.7, 0.3)
    # A paired Wilcoxon signed-rank test's exact null distribution only has
    # 2**n sign patterns (vs. an unpaired rank-sum test's much larger
    # C(2n, n)), so it needs more matched samples than 8 to clear the very
    # strict Holm-Bonferroni threshold here even for a clearly separated pair.
    _populate_runs(tmp_path, "full", n_seeds=16, method_means=method_means)

    comparisons = run_comparisons(tmp_path, scale="full")

    ball_vs_lora_brier = [
        c
        for c in comparisons
        if c.endpoint == "brier" and {c.method_a, c.method_b} == {"ball", "lora"}
    ]
    assert len(ball_vs_lora_brier) == len(DATASETS)
    assert all(c.significant for c in ball_vs_lora_brier)

    # Two baselines drawn from the same distribution should not be flagged.
    lora_vs_clora_brier = [
        c
        for c in comparisons
        if c.endpoint == "brier" and {c.method_a, c.method_b} == {"lora", "clora"}
    ]
    assert lora_vs_clora_brier == []  # not in our_vs_others_pairs (baseline vs baseline)


def test_required_test_runs_respects_floor(tmp_path):
    # Identical distributions everywhere: no detectable effect, so the huge
    # required-n estimate should be dominated by... actually a near-zero
    # variance still floors at MIN_TEST_RUNS when the estimated n is smaller.
    method_means = _all_methods_close_together()
    _populate_runs(tmp_path, "pilot", n_seeds=5, method_means=method_means)

    n = required_test_runs(tmp_path, delta=0.02)
    assert n >= 8
