# BALL
Replay-Free Bayesian Adaptors for Lifelong Learning and Uncertainty Quantification.


## Methods

In the paper we refer to our family of methods as BALL with the following variants:
- 2BALL: This is called `ball` in the codebase.
- 3BALL: This is called `tball` in the codebase.
- 3BALL_M: This is called `tball-mnd` in the codebase.


## Reproduce Experiments

To run with [uv](https://docs.astral.sh/uv/getting-started/installation/):
```
$ uv run main.py --help
Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  -c, --config FILE  Path to a jsonnet config file.  [required]
  -a, --args TEXT    Override config options using dotlist notation.
  -f, --force        Skip checking if repo is clean.
  --help             Show this message and exit.

Commands:
  hpsearch
  run
```

### Evaluation Runs

Evaluation runs train the model with the specified hyperparameters and evaluate the
model on the test set. Runs are logged to `log/{NAME}/{DATASET}/{METHOD}/{SEED}`

```bash
# uv run main.py -c configs/cifar100/ball.jsonnet run {name}  {seed}
$ uv run main.py -c configs/cifar100/ball.jsonnet run example 0
```

### Hyperparameter Search

Hyperparameter search uses [Optuna](https://optuna.org/) to optimize the hyperparameters
of the model. The hyperparameter search is ran against the validation set and evaluated
on an aggregate metric of accuracy and calibration. The results are stored in a database
specified by the `OPTUNA_STORAGE` environment variable.

```bash
$ export OPTUNA_STORAGE="sqlite:///example.db"
# uv run main.py -c configs/cifar100/ball.jsonnet hpsearch {study name}
$ uv run main.py -c configs/cifar100/ball.jsonnet hpsearch example-hp
```

The study is named `bayescl/example-hp/cifar100/ball` and can be accessed using the
Optuna dashboard:

```bash
$ uv run optuna-dashboard example.db
```

Once you have found the best hyperparameters, you may update the config files with the
best hyperparameters:

```sh
$ uv run script/hpupdate.py -s example-hp -d cifar100 -d core50 -m ball
```

<details>
<summary>hpupdate all</summary>

```sh
$ uv run script/hpupdate.py -s example-hp\
  -d cifar100 -d core50 -d imagenetr \
  -m ball -m clora -m ewc -m inflora -m lora -m rwalk -m sdlora -m tball -m tball-mnd
```

</details>

## Analysis

To collect the results of the hyperparameter search, you can run the following command:
```bash
$ uv run script/collect.py hpsearch example-hp example-hp.csv
```

To collect the results of the evaluation runs, you can run the following command:
```bash
$ uv run script/collect.py run example-run example-run.csv
```

You can find my runs and hyperparameter search metrics in the `results` directory.
