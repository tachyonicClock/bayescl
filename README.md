# BALL


## Algorithm

The BALL algorithm is implemented using PyTorch and Avalanche.

 * `bayescl/plugins/ball.py`: High level avalanche plugin orchestrating the BALL algorithm.
 * `bayescl/peft/ball.py`: Implementation of the Blob PEFT method.


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


## Virtual Environment

```bash
conda create -n ball python==3.12
conda activate ball
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
python main.py --hpsearch -c configs/${dataset}/${method}.yaml
```
This uses the optuna storage specified by `$OPTUNA_STORAGE` environment variable to
store the results. You can use a local sqlite file, or a remote database such as
MySQL or PostgreSQL. For example:
```bash
export OPTUNA_STORAGE="sqlite:///optuna.db"
python main.py --hpsearch -c configs/cifar100/01_linear.yaml --args epochs=1
```

### 3. Hyperparameter Search

```bash
ts -G 1 -m -L cifar_linea python main.py --hpsearch -c configs/cifar100/01_linear.yaml
ts -G 1 -m -L image_linea python main.py --hpsearch -c configs/imagenetr/01_linear.yaml
ts -G 1 -m -L domai_linea python main.py --hpsearch -c configs/domainnet/01_linear.yaml
ts -G 1 -m -L cifar_lora  python main.py --hpsearch -c configs/cifar100/02_lora.yaml
ts -G 1 -m -L image_lora  python main.py --hpsearch -c configs/imagenetr/02_lora.yaml
ts -G 1 -m -L domai_lora  python main.py --hpsearch -c configs/domainnet/02_lora.yaml
ts -G 1 -m -L cifar_ball  python main.py --hpsearch -c configs/cifar100/03_ball.yaml
ts -G 1 -m -L image_ball  python main.py --hpsearch -c configs/imagenetr/03_ball.yaml
ts -G 1 -m -L domai_ball  python main.py --hpsearch -c configs/domainnet/03_ball.yaml
ts -G 1 -m -L cifar_repla python main.py --hpsearch -c configs/cifar100/04_replay.yaml
ts -G 1 -m -L image_repla python main.py --hpsearch -c configs/imagenetr/04_replay.yaml
ts -G 1 -m -L domai_repla python main.py --hpsearch -c configs/domainnet/04_replay.yaml
ts -G 1 -m -L cifar_gdumb python main.py --hpsearch -c configs/cifar100/05_gdumb.yaml
ts -G 1 -m -L image_gdumb python main.py --hpsearch -c configs/imagenetr/05_gdumb.yaml
ts -G 1 -m -L domai_gdumb python main.py --hpsearch -c configs/domainnet/05_gdumb.yaml
ts -G 1 -m -L cifar_der   python main.py --hpsearch -c configs/cifar100/06_der.yaml
ts -G 1 -m -L image_der   python main.py --hpsearch -c configs/imagenetr/06_der.yaml
ts -G 1 -m -L domai_der   python main.py --hpsearch -c configs/domainnet/06_der.yaml
ts -G 1 -m -L cifar_joint python main.py --hpsearch -c configs/cifar100/07_joint.yaml
ts -G 1 -m -L image_joint python main.py --hpsearch -c configs/imagenetr/07_joint.yaml
ts -G 1 -m -L domai_joint python main.py --hpsearch -c configs/domainnet/07_joint.yaml
ts -G 1 -m -L cifar_rwalk python main.py --hpsearch -c configs/cifar100/08_rwalk.yaml
ts -G 1 -m -L image_rwalk python main.py --hpsearch -c configs/imagenetr/08_rwalk.yaml
ts -G 1 -m -L domai_rwalk python main.py --hpsearch -c configs/domainnet/08_rwalk.yaml

# Notify me when all jobs are completed
ts_notify 
```


```bash
# Copy conda env from lagerfield to local scratch
rsync -a --info=progress2 lagerfield.ecs.vuw.ac.nz:$ECS_SCRATCH/miniconda3/envs/bayescl $ECS_SCRATCH/miniconda3/envs
rsync -a --info=progress2 lagerfield.ecs.vuw.ac.nz:$DATASETS/cifar-100-python $DATASETS
rsync -a --info=progress2 lagerfield.ecs.vuw.ac.nz:$DATASETS/imagenet-r       $DATASETS
rsync -a --info=progress2 lagerfield.ecs.vuw.ac.nz:$DATASETS/domainnet        $DATASETS
pip install -r requirements.txt
```