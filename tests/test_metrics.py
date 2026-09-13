import math

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from bayescl.metrics.corruptions import corrupt, corruption_names
from bayescl.metrics.plugin import MetricsPlugin
from bayescl.metrics.results import ContinualLearningEvaluator


def test_nll_matches_hand_computed_value():
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    y = torch.tensor([0, 1])
    # log_softmax of a uniform 2-way logit is log(0.5) for both classes.
    expected = -math.log(0.5)
    assert ContinualLearningEvaluator.nll(logits, y) == pytest.approx(expected)


def test_auroc_perfect_separation_is_one():
    # id samples confidently predict class 0; ood samples are uniform (unconfident).
    id_logit = torch.tensor([[10.0, 0.0]] * 5)
    ood_logit = torch.tensor([[0.0, 0.0]] * 5)
    assert ContinualLearningEvaluator.auroc(id_logit, ood_logit) == pytest.approx(1.0)


def test_auroc_no_separation_is_half():
    torch.manual_seed(0)
    id_logit = torch.zeros(50, 2)
    ood_logit = torch.zeros(50, 2)
    assert ContinualLearningEvaluator.auroc(id_logit, ood_logit) == pytest.approx(0.5)


def _fill_evaluator(T: int = 3, C: int = 4, n: int = 16) -> ContinualLearningEvaluator:
    torch.manual_seed(0)
    ev = ContinualLearningEvaluator(T, C)
    for train_t in range(T):
        for test_t in range(T):
            ev.update(train_t, test_t, torch.randn(n, C), torch.randint(0, C, (n,)))
    return ev


def test_brier_seen_is_populated_and_finite():
    metrics, _ = _fill_evaluator().result()
    assert "brier_seen" in metrics
    assert np.all(np.isfinite(metrics["brier_seen"]))
    assert metrics["brier_seen_avg"] == pytest.approx(metrics["brier_seen"].mean())


def test_ece_seen_is_per_task_mean_not_pooled():
    T, C = 2, 3
    ev = ContinualLearningEvaluator(T, C)
    torch.manual_seed(0)

    logit0 = torch.randn(50, C)
    y0 = torch.randint(0, C, (50,))
    ev.update(0, 0, logit0, y0)
    # (0, 1): task 1 evaluated zero-shot before it's been trained on -- every
    # checkpoint evaluates the full test stream, seen and future tasks alike.
    ev.update(0, 1, torch.randn(30, C), torch.randint(0, C, (30,)))

    # Deliberately unequal per-task sample counts, so pooling samples before
    # computing ECE (nonlinear due to binning) diverges from averaging each
    # seen task's own ECE.
    logit1_0 = torch.randn(5, C)
    y1_0 = torch.randint(0, C, (5,))
    logit1_1 = torch.randn(200, C)
    y1_1 = torch.randint(0, C, (200,))
    ev.update(1, 0, logit1_0, y1_0)
    ev.update(1, 1, logit1_1, y1_1)

    metrics, _ = ev.result()

    expected_seen_1 = np.mean(
        [
            ContinualLearningEvaluator.ece(logit1_0, y1_0),
            ContinualLearningEvaluator.ece(logit1_1, y1_1),
        ]
    )
    pooled_seen_1 = ContinualLearningEvaluator.ece(
        torch.cat([logit1_0, logit1_1]), torch.cat([y1_0, y1_1])
    )

    assert metrics["ece_seen"][1] == pytest.approx(expected_seen_1)
    # Regression guard: pooling and per-task averaging genuinely differ here,
    # so this would catch a silent revert to the pooled approach.
    assert metrics["ece_seen"][1] != pytest.approx(pooled_seen_1)


def test_brier_handles_batches_missing_some_classes():
    # sklearn's brier_score_loss infers the class set from y_true alone unless
    # told otherwise, which breaks as soon as a batch doesn't contain every
    # class -- always true for brier_seen early in training (only a few
    # classes seen so far). Regression test for that fix.
    logits = torch.randn(5, 6)
    y_true = torch.tensor([0, 1, 0, 1, 0])  # only 2 of 6 classes present
    value = ContinualLearningEvaluator.brier(logits, y_true)
    assert np.isfinite(value)


def test_nll_seen_is_populated_and_finite():
    metrics, _ = _fill_evaluator().result()
    assert np.all(np.isfinite(metrics["nll_seen"]))
    assert np.all(np.isfinite(metrics["nll_all"]))


def test_auroc_future_excludes_final_checkpoint():
    T = 4
    metrics, _ = _fill_evaluator(T=T).result()
    assert metrics["auroc_future"].shape == (T - 1,)
    assert np.all((metrics["auroc_future"] >= 0) & (metrics["auroc_future"] <= 1))


def test_auroc_future_absent_for_single_task():
    metrics, _ = _fill_evaluator(T=1).result()
    assert "auroc_future" not in metrics


def test_record_ood_produces_per_dataset_auroc():
    T, C = 3, 4
    ev = _fill_evaluator(T=T, C=C)
    for t in range(T):
        ev.record_ood("svhn", t, torch.randn(10, C))
    metrics, _ = ev.result()
    assert metrics["auroc_svhn"].shape == (T,)
    assert 0 <= metrics["auroc_svhn_avg"] <= 1
    # An OOD dataset never recorded shouldn't show up at all.
    assert "auroc_cifar10" not in metrics


def test_record_shift_produces_ece_and_ace_per_severity():
    T, C = 3, 4
    ev = _fill_evaluator(T=T, C=C)
    for t in range(T):
        ev.record_shift(3, t, torch.randn(20, C), torch.randint(0, C, (20,)))
    metrics, _ = ev.result()
    assert metrics["ece_shift_3"].shape == (T,)
    assert metrics["ace_shift_3"].shape == (T,)
    assert 0 <= metrics["ece_shift_3_avg"] <= 1
    assert 0 <= metrics["ace_shift_3_avg"] <= 1
    # A severity never recorded shouldn't show up at all.
    assert "ece_shift_1" not in metrics


def test_shift_and_ood_absent_when_never_recorded():
    metrics, _ = _fill_evaluator().result()
    assert not [k for k in metrics if k.startswith("ece_shift_")]
    assert not [k for k in metrics if k.startswith("auroc_") and k != "auroc_future" and k != "auroc_future_avg"]


class _FakeStrategy:
    def __init__(self, logits: torch.Tensor, y: torch.Tensor):
        self.mbatch = (None, y, None)
        self.mb_output = logits


def test_metrics_plugin_normal_recording():
    plugin = MetricsPlugin(num_tasks=2, num_classes=3)
    plugin.before_training_exp(None)
    plugin.after_training_exp(None)
    plugin.before_eval(None)
    strategy = _FakeStrategy(torch.randn(4, 3), torch.randint(0, 3, (4,)))
    plugin.after_eval_iteration(strategy)
    assert (0, 0) in plugin.evaluator._y_logit


def test_metrics_plugin_capture_isolates_from_evaluator():
    plugin = MetricsPlugin(num_tasks=2, num_classes=3)
    plugin.before_training_exp(None)
    plugin.after_training_exp(None)
    plugin.before_eval(None)

    captured = []
    strategy = _FakeStrategy(torch.randn(5, 3), torch.randint(0, 3, (5,)))
    with plugin.capture(lambda logits, y: captured.append((logits, y))):
        plugin.after_eval_iteration(strategy)

    assert len(captured) == 1
    assert captured[0][0].shape == (5, 3)
    # The main evaluator must not have recorded anything during capture.
    assert plugin.evaluator._y_logit == {}

    # After the context manager exits, normal recording resumes.
    strategy2 = _FakeStrategy(torch.randn(2, 3), torch.randint(0, 3, (2,)))
    plugin.after_eval_iteration(strategy2)
    assert (0, 0) in plugin.evaluator._y_logit


def test_corruption_names_and_corrupt_roundtrip():
    from PIL import Image

    img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
    names = corruption_names()
    assert len(names) == 13
    for name in names:
        out = corrupt(img, name, severity=2)
        assert out.size == img.size
        assert out.mode == "RGB"


def test_corrupt_rejects_out_of_range_severity():
    from PIL import Image

    img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
    with pytest.raises(ValueError):
        corrupt(img, corruption_names()[0], severity=0)
    with pytest.raises(ValueError):
        corrupt(img, corruption_names()[0], severity=6)


def test_shifted_tensor_dataset_preserves_shape_and_targets():
    from bayescl.benchmark import ShiftedTensorDataset

    torch.manual_seed(0)
    x = torch.rand(6, 3, 32, 32)
    y = torch.randint(0, 4, (6,))

    class Toy(Dataset):
        targets = y.tolist()

        def __len__(self):
            return len(y)

        def __getitem__(self, idx):
            return x[idx], int(y[idx]), 0

    shifted = ShiftedTensorDataset(Toy(), severity=2, standardize=False)
    assert len(shifted) == 6
    assert shifted.targets == y.tolist()
    # Only (x, y) -- the task label is dropped, since wrapping this dataset
    # through create_multi_dataset_generic_benchmark adds a fresh one.
    out_x, out_y = shifted[0]
    assert out_x.shape == (3, 32, 32)
    assert out_y == int(y[0])
