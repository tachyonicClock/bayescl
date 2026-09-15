from pathlib import Path

import pytest

from bayescl.config import ExperimentConfig
from bayescl.data import benchmark as benchmark_module


def _make_config(dataset: str, seed: int) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset,
        n_tasks=5,
        shuffle=True,
        dataset_root=Path("/unused"),
        standardize=True,
        validation=False,
        scale="pilot",
        backbone_name="unused",
        freeze_backbone=True,
        adapter_filter=".*",
        head_module="model.classifier",
        epochs=1,
        train_mb_size=1,
        eval_mb_size=1,
        num_workers=0,
        run_dir=Path("/unused"),
        seed=seed,
        device="cpu",
    )


@pytest.mark.parametrize(
    "dataset, split_fn_name",
    [
        ("CIFAR100", "SplitCIFAR100"),
        ("ImageNetR", "SplitImageNetR"),
        ("DomainNet", "SplitDomainNet"),
        ("CUB200_2011", "SplitCUB200_2011"),
    ],
)
def test_get_benchmark_threads_config_seed(monkeypatch, dataset, split_fn_name):
    """``get_benchmark`` must pass ``config.seed`` through to the dataset
    constructor's ``nc_benchmark(..., seed=...)`` call.

    Regression test: ``Experiment._val_stream()`` builds a second benchmark
    (for early-stopping validation) via ``get_benchmark(replace(config,
    validation=True))`` after model/adapter construction has already
    consumed RNG state. Without an explicit seed, ``nc_benchmark`` shuffles
    task order using whatever the ambient PyTorch RNG state happens to be,
    so that second benchmark's task order silently diverged from the first
    -- early-stopping was scoring each task against a different task's
    classes. Passing ``seed=config.seed`` explicitly makes shuffling
    independent of call order/ambient state, so both benchmarks (same seed)
    always agree.
    """
    captured = {}

    def fake_split(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(benchmark_module, split_fn_name, fake_split)

    config = _make_config(dataset, seed=7)
    benchmark_module.get_benchmark(config)

    assert captured.get("seed") == 7


def test_val_stream_config_preserves_seed_for_early_stopping():
    """``replace(config, validation=True)`` -- what ``_val_stream`` uses to
    build the early-stopping benchmark -- must not change ``seed``, or the
    two benchmarks would diverge even with seed threading in place."""
    from dataclasses import replace

    config = _make_config("CIFAR100", seed=7)
    val_config = replace(config, validation=True)

    assert val_config.seed == config.seed
