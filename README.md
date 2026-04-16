# BALL
Replay-Free Bayesian Adaptors for Lifelong Learning and Uncertainty Quantification.

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

## 

```bash
uv run main.py -c configs/cifar100/ball.jsonnet run
```
