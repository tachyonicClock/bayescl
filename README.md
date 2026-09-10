# BALL
Replay-Free Bayesian Adaptors for Lifelong Learning and Uncertainty Quantification.


## Methods

In the paper we refer to our family of methods as BALL with the following variants:
- 2BALL: This is called `ball` in the codebase.
- 3BALL: This is called `tball` in the codebase.
- 3BALL_M: This is called `tball-mnd` in the codebase.

Each method ("arm") is a single dataclass in `bayescl/methods/<name>/_arm.py`
holding its hyperparameters, a `suggest_config(trial, base)` that samples them
from an Optuna trial, and builder hooks that assemble a runnable `Experiment`.
Variants such as `tball-mnd` are `@register`ed subclasses that only change field
defaults. All non-method configuration — datasets, the `pilot` / `full` budgets,
and the resolved `ExperimentConfig` — lives in `bayescl/config.py`.


## Reproduce Experiments

To run with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```
$ uv run main.py --help
Usage: main.py [OPTIONS] COMMAND [ARGS]...

Commands:
  tune  Search hyperparameters for METHOD on DATASET at the given SCALE.
  test  Evaluate METHOD's best tuned config on DATASET over `n_seeds` seeds.
```

Both commands take the same positional arguments:

```
main.py <tune|test> <pilot|full> <dataset> <method>
```

- `scale` — `pilot` is a fast smoke run (few trials, few epochs, one seed);
  `full` is the paper-quality budget (see `bayescl/config.py`).
- `dataset` — one of `cifar100`, `core50`, `imagenetr`.
- `method` — one of `ball`, `clora`, `ewc`, `inflora`, `lora`, `rwalk`,
  `sdlora`, `tball`, `tball-mnd`.

Set `$DATASETS` to your datasets directory (or pass `--dataset-path`).

### 1. Tune

```bash
$ uv run main.py tune full cifar100 ball
```

Runs an Optuna study against a validation split and writes:

```
runs/tune/full/cifar100/ball/<RUNID>/
    results.jsonl   # one line per trial: {trial, arm, params, acc, ece, score, state}
    meta.json       # git provenance, scale knobs, best trial
    trial_0000/ ...  # per-trial Experiment output (TensorBoard, metrics.pkl, ...)
```

`<RUNID>` is a `%Y-%m-%d_%H-%M-%S` timestamp. The objective maximises
`score = 0.5 * (accuracy + (1 - ECE))`.

Add `--sqlite` to also write `runs/.../optuna.db` for the dashboard:

```bash
$ uv run optuna-dashboard runs/tune/full/cifar100/ball/<RUNID>/optuna.db
```

### 2. Test

```bash
$ uv run main.py test full cifar100 ball
```

Finds the most recent `runs/tune/full/cifar100/ball/*/` (or use `--from-tune
PATH`), picks the trial with the best `score`, reconstructs the arm, and runs it
for `n_seeds` seeds against the test set:

```
runs/test/full/cifar100/ball/<RUNID>/
    results.jsonl   # one line per seed: {seed, acc, ece, score}
    meta.json
    seed_00/ ...
```

> Arm defaults are generic (`lr=1e-3`). `test` only produces meaningful numbers
> after a `full` `tune` for that (dataset, method) pair.


## Analysis

Collect run results into a CSV:

```bash
$ uv run script/collect.py ./runs results.csv --stage test
$ uv run script/collect.py ./runs trials.csv --stage tune
```

You can find my runs and hyperparameter search metrics in the `results`
directory.
