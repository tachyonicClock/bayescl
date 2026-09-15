import pytest
import torch

from bayescl.base import ConvergenceError
from bayescl.experiment import BrierEarlyStopping


class _FakeStrategy:
    def __init__(self, train_epochs=None):
        self.model = torch.nn.Linear(2, 2)
        self.stopped = False
        self.train_epochs = train_epochs

    def stop_training(self):
        self.stopped = True


def _make_plugin(monkeypatch, briers, eval_every=2, patience=2, strict=False):
    calls = iter(briers)

    def fake_eval_and_capture(strategy, val_experience, loader_kwargs):
        return torch.zeros(1, 2), torch.zeros(1, dtype=torch.long)

    monkeypatch.setattr(
        "bayescl.metrics.results.ContinualLearningEvaluator.brier",
        staticmethod(lambda logits, y: next(calls)),
    )
    return BrierEarlyStopping(
        fake_eval_and_capture,
        val_stream=[None],
        loader_kwargs={},
        eval_every=eval_every,
        patience=patience,
        strict=strict,
    )


def _run_epochs(plugin, strategy, n_epochs):
    """Simulate ``n_epochs`` real training epochs against ``plugin``."""
    for _ in range(n_epochs):
        plugin.before_training_epoch(strategy)
        if strategy.stopped:
            return
        plugin.after_training_epoch(strategy)


def test_stops_after_patience_non_improving_checks(monkeypatch):
    strategy = _FakeStrategy()
    # baseline (before any epoch) = 1.0, improves at the epoch-2 checkpoint,
    # then plateaus at every subsequent checkpoint.
    briers = [1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    plugin = _make_plugin(monkeypatch, briers, eval_every=2, patience=2)

    plugin.before_training_exp(strategy)
    _run_epochs(plugin, strategy, 10)

    assert strategy.stopped
    assert plugin._epochs_without_improvement == plugin.patience
    assert plugin._best_brier == 0.8


def test_runs_to_completion_when_still_improving(monkeypatch):
    strategy = _FakeStrategy()
    briers = [1.0, 0.9, 0.8, 0.7, 0.6]
    plugin = _make_plugin(monkeypatch, briers, eval_every=2, patience=2)

    plugin.before_training_exp(strategy)
    _run_epochs(plugin, strategy, 8)

    assert not strategy.stopped
    assert plugin._best_brier == 0.6


def test_after_training_exp_restores_best_checkpoint_even_without_early_stop(
    monkeypatch,
):
    strategy = _FakeStrategy()
    # improves at the epoch-2 checkpoint, then gets worse at epoch-4, but
    # never accumulates enough non-improving checks to trigger stop_training.
    briers = [1.0, 0.5, 0.9]
    plugin = _make_plugin(monkeypatch, briers, eval_every=2, patience=10)

    plugin.before_training_exp(strategy)
    _run_epochs(plugin, strategy, 2)  # checkpoint at brier=0.5
    best_checkpoint = {k: v.clone() for k, v in strategy.model.state_dict().items()}

    with torch.no_grad():
        for p in strategy.model.parameters():
            p.add_(1.0)  # simulate further training that drifts the weights

    _run_epochs(plugin, strategy, 2)  # checkpoint at brier=0.9, worse

    assert not strategy.stopped
    plugin.after_training_exp(strategy)  # should restore the epoch-2 checkpoint

    for k, v in strategy.model.state_dict().items():
        assert torch.equal(v, best_checkpoint[k])


def test_strict_raises_convergence_error_when_epoch_budget_exhausted(monkeypatch):
    strategy = _FakeStrategy(train_epochs=6)
    # improves at every single checkpoint -- patience never accumulates, so
    # training only stops because it runs out of its 6-epoch budget.
    briers = [1.0, 0.9, 0.8, 0.7]
    plugin = _make_plugin(monkeypatch, briers, eval_every=2, patience=2, strict=True)

    plugin.before_training_exp(strategy)
    _run_epochs(plugin, strategy, 6)

    assert not strategy.stopped  # never hit patience, ran the full budget
    with pytest.raises(ConvergenceError):
        plugin.after_training_exp(strategy)


def test_non_strict_does_not_raise_when_epoch_budget_exhausted(monkeypatch):
    strategy = _FakeStrategy(train_epochs=6)
    briers = [1.0, 0.9, 0.8, 0.7]
    plugin = _make_plugin(monkeypatch, briers, eval_every=2, patience=2, strict=False)

    plugin.before_training_exp(strategy)
    _run_epochs(plugin, strategy, 6)

    assert not strategy.stopped
    plugin.after_training_exp(strategy)  # should not raise


def test_strict_does_not_raise_when_early_stopping_triggered_first(monkeypatch):
    strategy = _FakeStrategy(train_epochs=100)
    briers = [1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    plugin = _make_plugin(monkeypatch, briers, eval_every=2, patience=2, strict=True)

    plugin.before_training_exp(strategy)
    _run_epochs(plugin, strategy, 10)

    assert strategy.stopped  # converged well before the 100-epoch budget
    plugin.after_training_exp(strategy)  # should not raise
