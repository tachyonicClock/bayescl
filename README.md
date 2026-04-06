# BALL

## Overview

* `bayescl`: Source code implementing BALL and other experiments.
* `metadata`: Datasplits for DomainNet dataset.
* `configs`: Configuration files defining hyperparameters search spaces and final
  configurations for each dataset and method.
    * Explanations of each configuration parameter can be found in `bayescl/config.py`.
* `main.py`: Main entry point to run experiments.
* See [LLM.md](LLM.md) for details on the use of large language models in this project.


## Algorithm

The BALL algorithm is implemented using [PyTorch](https://pytorch.org) and [Avalanche](https://avalanche.continualai.org/) (continual learning library).
The main components are:

 * `bayescl/plugins/ball.py`: High level Avalanche plugin orchestrating the BALL
   algorithm.
    * Add KL divergence loss to the training loss.
    * Sets the prior of the new task to the posterior of the previous task.
 * `bayescl/peft/_ball/layer.py`: Implementation of the Bayesian Adaptor for Lifelong
   Learning (BALL) layer.
 * `bayescl/peft/_ball/factory.py`: Factory to create BALL layers and replace linear
   layers in a model.
 * `bayescl/vbnn.py`: Implementation of variational Bayesian neural networks.
    * `get_model_kl_loss`: Compute the KL divergence of a model with Bayesian layers.
 * `bayescl/experiment.py`: Is a big constructor that assembles an Avalanche strategy
   with the appropriate plugins, loggers, evaluators, and model.
    * `_add_peft_adapters`: Function that adds BALL(/PEFT) layers to a model and setups
      the plugin.


## Datasets

* `CIFAR100` (downloads automatically): https://www.cs.toronto.edu/~kriz/cifar.html
* `DomainNet` (manual setup): https://ai.bu.edu/M3SDA/
* `ImageNet-R` (manual setup): https://github.com/hendrycks/imagenet-r

The codebase by default looks for datasets in the path specified by `$DATASETS`
environment variable. The expected folder structure is as follows:
```text
$DATASETS
├── cifar-100-python
│   ├── meta
│   ├── test
│   └── train
├── domainnet
│   ├── clipart
│   ├── domainnet_test.yaml
│   ├── domainnet_train.yaml
│   ├── infograph
│   ├── painting
│   ├── quickdraw
│   ├── README.md
│   ├── real
│   └── sketch
└── imagenet-r
    ├── README.txt
    ├── test
    └── train
```

The datasplit for `DomainNet` is provided in `metadata` folder. The split is identical
to the one used by
[InfLoRA](https://github.com/liangyanshuo/InfLoRA/tree/e08b00edd54f2f10cf2f9826eae7d44fdcb6354b/dataloaders/splits).
Decompress and move the files to `$DATASETS/domainnet/`:
```bash
gzip -d < metadata/domainnet_train.yaml.gz > $DATASETS/domainnet/domainnet_train.yaml
gzip -d < metadata/domainnet_test.yaml.gz  > $DATASETS/domainnet/domainnet_test.yaml
```

## Results

We save the results of each experiment in `log/` folder by default. The results are organized as follows:

* `avalanche_results.pkl`: Pickle containing a list of collected avalanche metrics.
* `config.json`: Configuration file used for the experiment.
* `metrics.pkl`: Pickled dictionary containing detailed metrics including
  accuracy matrices and expected calibration error.
* `events.out.tfevents.1762118680.*.3634449.0`: Tensorboard log file.
* `raw_data.pkl`: Pickled logits/true labels collected during evaluation.
* `checkpoint-*.pth`: Model checkpoint saved after each task (if enabled).

## Virtual Environment

```bash
conda create -n bayescl python==3.12
conda activate bayescl
pip install -r requirements.txt
```

## Run

To run a specific method on a specific dataset, use:
```bash
python main.py -c configs/${dataset}/${method}.yaml --args epochs=1
```

To inspect the tensorboard logs:
```bash
tensorboard --logdir log
```

### Override Configurations

Any configuration in the config file can be overridden using `--args` flag. For example:

```bash
python main.py -c configs/${dataset}/${method}.yaml --args epochs=1
```
Take a look at `bayescl/config.py` for the documented configuration object.
It is likely that not all configurations are valid.

### Hyperparameter Search

The config files are configured with the best hyperparameters found during our
hyperparameter search. The files `configs/base/method/${method}.yaml` contain the
hyperparameter search space and configurations common across datasets.

To reproduce the hyperparameters search, run with `--hpsearch` flag:
```bash
python main.py -c configs/${dataset}/${method}.yaml hpsearch
```
This uses the optuna storage specified by `$OPTUNA_STORAGE` environment variable to
store the results. You can use a local sqlite file, or a remote database such as
MySQL or PostgreSQL. For example:
```bash
export OPTUNA_STORAGE="sqlite:///optuna.db"
python main.py --hpsearch -c configs/cifar100/01_linear.yaml --args epochs=1
```

### Hardware

* The code has been tested with CUDA 12.9 and Experiments were conducted on various GPUs with CUDA
12.9: Quadro RTX 6000, L4, RTX A5000, and  RTX A6000.
* A GPU with at least 20GB of memory is needed for the default batch sizes.
* 16GB of RAM.
* 6 core CPU for data loading.
* Linux.

## Script (TODO: Remove)

```bash
ts -G 1 -L C100.RWLK   notirun.sh python main.py        -c configs/cifar100/rwalk.jsonnet hpsearch hp
ts -G 1 -L C100.EWC    notirun.sh python main.py          -c configs/cifar100/ewc.jsonnet hpsearch hp
ts -G 1 -L C100.LORA   notirun.sh python main.py         -c configs/cifar100/lora.jsonnet hpsearch hp
ts -G 1 -L C100.CLORA  notirun.sh python main.py        -c configs/cifar100/clora.jsonnet hpsearch hp
ts -G 1 -L C100.SDLORA notirun.sh python main.py       -c configs/cifar100/sdlora.jsonnet hpsearch hp
ts -G 1 -L C100.INFLO  notirun.sh python main.py      -c configs/cifar100/inflora.jsonnet hpsearch hp
ts -G 1 -L C100.BALL   notirun.sh python main.py         -c configs/cifar100/ball.jsonnet hpsearch hp
ts -G 1 -L C100.TBALL  notirun.sh python main.py        -c configs/cifar100/tball.jsonnet hpsearch hp
ts -G 1 -L C100.TBMND  notirun.sh python main.py    -c configs/cifar100/tball-mnd.jsonnet hpsearch hp

ts -G 1 -L C50.RWLK    notirun.sh python main.py          -c configs/core50/rwalk.jsonnet hpsearch hp
ts -G 1 -L C50.EWC     notirun.sh python main.py            -c configs/core50/ewc.jsonnet hpsearch hp
ts -G 1 -L C50.LORA    notirun.sh python main.py           -c configs/core50/lora.jsonnet hpsearch hp
ts -G 1 -L C50.CLORA   notirun.sh python main.py          -c configs/core50/clora.jsonnet hpsearch hp
ts -G 1 -L C50.SDLORA  notirun.sh python main.py         -c configs/core50/sdlora.jsonnet hpsearch hp
ts -G 1 -L C50.INFLO   notirun.sh python main.py        -c configs/core50/inflora.jsonnet hpsearch hp
ts -G 1 -L C50.BALL    notirun.sh python main.py           -c configs/core50/ball.jsonnet hpsearch hp
ts -G 1 -L C50.TBALL   notirun.sh python main.py          -c configs/core50/tball.jsonnet hpsearch hp
ts -G 1 -L C50.TBMND   notirun.sh python main.py      -c configs/core50/tball-mnd.jsonnet hpsearch hp

ts -G 1 -L INR.RWLK    notirun.sh python main.py       -c configs/imagenetr/rwalk.jsonnet hpsearch hp
ts -G 1 -L INR.EWC     notirun.sh python main.py         -c configs/imagenetr/ewc.jsonnet hpsearch hp
ts -G 1 -L INR.LORA    notirun.sh python main.py        -c configs/imagenetr/lora.jsonnet hpsearch hp
ts -G 1 -L INR.CLORA   notirun.sh python main.py       -c configs/imagenetr/clora.jsonnet hpsearch hp
ts -G 1 -L INR.SDLORA  notirun.sh python main.py      -c configs/imagenetr/sdlora.jsonnet hpsearch hp
ts -G 1 -L INR.INFLO   notirun.sh python main.py     -c configs/imagenetr/inflora.jsonnet hpsearch hp
ts -G 1 -L INR.BALL    notirun.sh python main.py        -c configs/imagenetr/ball.jsonnet hpsearch hp
ts -G 1 -L INR.TBALL   notirun.sh python main.py       -c configs/imagenetr/tball.jsonnet hpsearch hp
ts -G 1 -L INR.TBMND   notirun.sh python main.py   -c configs/imagenetr/tball-mnd.jsonnet hpsearch hp
```

## Energy Analysis

```bash
ts -G 1 -L Zrwalk  notirun.sh python main.py -f -a epochs=5 -c configs/cifar100/rwalk.jsonnet     run --zeus zeus 0
ts -G 1 -L Zewc    notirun.sh python main.py -f -a epochs=5 -c configs/cifar100/ewc.jsonnet       run --zeus zeus 0
ts -G 1 -L Zlora   notirun.sh python main.py -f -a epochs=5 -c configs/cifar100/lora.jsonnet      run --zeus zeus 0
ts -G 1 -L Zclora  notirun.sh python main.py -f -a epochs=5 -c configs/cifar100/clora.jsonnet     run --zeus zeus 0
ts -G 1 -L Zsdlora notirun.sh python main.py -f -a epochs=5 -c configs/cifar100/sdlora.jsonnet    run --zeus zeus 0
ts -G 1 -L Zinflor notirun.sh python main.py -f -a epochs=5 -c configs/cifar100/inflora.jsonnet   run --zeus zeus 0
ts -G 1 -L Zball   notirun.sh python main.py -f -a epochs=5 -c configs/cifar100/ball.jsonnet      run --zeus zeus 0
ts -G 1 -L Ztball  notirun.sh python main.py -f -a epochs=5 -c configs/cifar100/tball.jsonnet     run --zeus zeus 0
ts -G 1 -L Ztballm notirun.sh python main.py -f -a epochs=5 -c configs/cifar100/tball-mnd.jsonnet run --zeus zeus 0
ts -G 1 -L Znolora notirun.sh python main.py -f -a epochs=5 -c configs/cifar100/no-lora.jsonnet run --zeus zeus 0
```


## Sensitivity Analysis

```sh
ts -G 1 -L cifar_ball_sens notirun.sh python main.py \
  -c configs/cifar100-sensitivity/ball-beta.jsonnet hpsearch 01-sensitivity-beta
ts -G 1 -L cifar_tball_sens notirun.sh python main.py \
  -c configs/cifar100-sensitivity/tball-beta.jsonnet hpsearch 01-sensitivity-beta
ts -G 1 -L cifar_tball_mnd_sens notirun.sh python main.py \
  -c configs/cifar100-sensitivity/tball-mnd-beta.jsonnet hpsearch 01-sensitivity-beta

ts -G 1 -L cifar_ball_rank notirun.sh python main.py \
  -c configs/cifar100-sensitivity/ball-rank.jsonnet hpsearch 01-sensitivity-rank
ts -G 1 -L cifar_tball_rank notirun.sh python main.py \
  -c configs/cifar100-sensitivity/tball-rank.jsonnet hpsearch 01-sensitivity-rank
ts -G 1 -L cifar_tball_mnd_rank notirun.sh python main.py \
  -c configs/cifar100-sensitivity/tball-mnd-rank.jsonnet hpsearch 01-sensitivity-rank

ts -G 1 -L cifar_ball_sens notirun.sh python main.py \
  -c configs/cifar100-sensitivity/ball-samples.jsonnet hpsearch 01-sensitivity-samples
ts -G 1 -L cifar_tball_sens notirun.sh python main.py \
  -c configs/cifar100-sensitivity/tball-samples.jsonnet hpsearch 01-sensitivity-samples
ts -G 1 -L cifar_tball_mnd_sens notirun.sh python main.py \
  -c configs/cifar100-sensitivity/tball-mnd-samples.jsonnet hpsearch 01-sensitivity-samples
```

