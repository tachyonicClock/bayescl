# bayescl

```bash
python main.py -c configs/${dataset}/${method}.yaml
```

## Methodology

```
mkdir -p /local/scratch/antonlee/log/bayescl
```

```
conda create -n bayescl python==3.12
```

```bash
# Copy conda env from lagerfield to local scratch
rsync -a --info=progress2 lagerfield.ecs.vuw.ac.nz:$ECS_SCRATCH/miniconda3/envs/bayescl $ECS_SCRATCH/miniconda3/envs
rsync -a --info=progress2 lagerfield.ecs.vuw.ac.nz:$DATASETS/cifar-100-python $DATASETS
conda activate bayescl
```


### 1. Setup Environment
```bash
cd /home/antonlee/github.com/tachyonicClock/bayescl_frozen
conda activate bayescl
pip install -r requirements.txt
```

### 2. Setup Task Spooler
```
ts --set_gpu_free_perc=97
ts -S 3 #get/set the number of max simultaneous jobs of the server.
```

### 3. Hyperparameter Search

#### iCIFAR100/10
```bash
ts -G 1 -m -L 01_linear  python main.py --hpsearch --configs configs/cifar100/01_linear.yaml
ts -G 1 -m -L 02_lora    python main.py --hpsearch --configs configs/cifar100/02_lora.yaml
ts -G 1 -m -L 03_blob    python main.py --hpsearch --configs configs/cifar100/03_blob.yaml
ts -G 1 -m -L 04_clora   python main.py --hpsearch --configs configs/cifar100/04_clora.yaml
ts -G 1 -m -L 05_inflora python main.py --hpsearch --configs configs/cifar100/05_inflora.yaml
```

#### iImageNet-R200/10

```bash
ts -G 1 -m -L 01_linear  python main.py --hpsearch --configs configs/imagenetr/01_linear.yaml
ts -G 1 -m -L 02_lora    python main.py --hpsearch --configs configs/imagenetr/02_lora.yaml
ts -G 1 -m -L 03_blob    python main.py --hpsearch --configs configs/imagenetr/03_blob.yaml
ts -G 1 -m -L 04_clora   python main.py --hpsearch --configs configs/imagenetr/04_clora.yaml
ts -G 1 -m -L 05_inflora python main.py --hpsearch --configs configs/imagenetr/05_inflora.yaml
```

#### iDomainNet345/5
```bash
ts -G 1 -m -L 01_linear  python main.py --hpsearch --configs configs/imagenetr/01_linear.yaml
ts -G 1 -m -L 02_lora    python main.py --hpsearch --configs configs/imagenetr/02_lora.yaml
ts -G 1 -m -L 03_blob    python main.py --hpsearch --configs configs/imagenetr/03_blob.yaml
ts -G 1 -m -L 04_clora   python main.py --hpsearch --configs configs/imagenetr/04_clora.yaml
ts -G 1 -m -L 05_inflora python main.py --hpsearch --configs configs/imagenetr/05_inflora.yaml
```

### 5. Update Configs with Best Hyperparameters
You should update `configs/*/*.yaml` to reflect the best hyperparameters found.

### 6. Run Experiments with Best Hyperparameters

#### iCIFAR100/10
```bash
ts -G 1 -m -L 01_linear  python main.py --repeat=5 --configs configs/cifar100/01_linear.yaml
ts -G 1 -m -L 02_lora    python main.py --repeat=5 --configs configs/cifar100/02_lora.yaml
ts -G 1 -m -L 03_blob    python main.py --repeat=5 --configs configs/cifar100/03_blob.yaml
ts -G 1 -m -L 04_clora   python main.py --repeat=5 --configs configs/cifar100/04_clora.yaml
ts -G 1 -m -L 05_inflora python main.py --repeat=5 --configs configs/cifar100/05_inflora.yaml
```

#### iImageNet-R200/10
```bash
ts -G 1 -m -L 01_linear  python main.py --repeat=5 --configs configs/imagenetr/01_linear.yaml
ts -G 1 -m -L 02_lora    python main.py --repeat=5 --configs configs/imagenetr/02_lora.yaml
ts -G 1 -m -L 03_blob    python main.py --repeat=5 --configs configs/imagenetr/03_blob.yaml
ts -G 1 -m -L 04_clora   python main.py --repeat=5 --configs configs/imagenetr/04_clora.yaml
ts -G 1 -m -L 05_inflora python main.py --repeat=5 --configs configs/imagenetr/05_inflora.yaml
```

#### iDomainNet345/5
TODO
