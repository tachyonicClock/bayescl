# bayescl

```bash
python main.py -c configs/${dataset}/${method}.yaml
```

## Hyperparameter Search

```bash
cd /home/antonlee/github.com/tachyonicClock/bayescl_frozen
conda activate bayescl
pip install -r requirements.txt
ts --set_gpu_free_perc=97
ts -S 3 #get/set the number of max simultaneous jobs of the server.

# HYPERPARAMETER SEARCH PHASE ----------------------------------------------------------
# iCIFAR100/10
ts -G 1 -m -L 01_linear  python main.py --hpsearch --configs configs/cifar100/01_linear.yaml
ts -G 1 -m -L 02_lora    python main.py --hpsearch --configs configs/cifar100/02_lora.yaml
ts -G 1 -m -L 03_blob    python main.py --hpsearch --configs configs/cifar100/03_blob.yaml
ts -G 1 -m -L 04_clora   python main.py --hpsearch --configs configs/cifar100/04_clora.yaml
ts -G 1 -m -L 05_inflora python main.py --hpsearch --configs configs/cifar100/05_inflora.yaml

# iImageNet-R200/10
ts -G 1 -m -L 01_linear  python main.py --hpsearch --configs configs/imagenetr/01_linear.yaml
ts -G 1 -m -L 02_lora    python main.py --hpsearch --configs configs/imagenetr/02_lora.yaml
ts -G 1 -m -L 03_blob    python main.py --hpsearch --configs configs/imagenetr/03_blob.yaml
ts -G 1 -m -L 04_clora   python main.py --hpsearch --configs configs/imagenetr/04_clora.yaml
ts -G 1 -m -L 05_inflora python main.py --hpsearch --configs configs/imagenetr/05_inflora.yaml

# iDomainNet345/5
ts -G 1 -m -L 01_linear  python main.py --hpsearch --configs configs/imagenetr/01_linear.yaml
ts -G 1 -m -L 02_lora    python main.py --hpsearch --configs configs/imagenetr/02_lora.yaml
ts -G 1 -m -L 03_blob    python main.py --hpsearch --configs configs/imagenetr/03_blob.yaml
ts -G 1 -m -L 04_clora   python main.py --hpsearch --configs configs/imagenetr/04_clora.yaml
ts -G 1 -m -L 05_inflora python main.py --hpsearch --configs configs/imagenetr/05_inflora.yaml

# You should update ./configs/*/*.yaml to reflect the best hyperparameters found.

# EXPERIMENT PHASE ---------------------------------------------------------------------
# CIFAR100
ts -G 1 -m -L 01_linear  python main.py --repeat=5 --configs configs/cifar100/01_linear.yaml
ts -G 1 -m -L 02_lora    python main.py --repeat=5 --configs configs/cifar100/02_lora.yaml
ts -G 1 -m -L 03_blob    python main.py --repeat=5 --configs configs/cifar100/03_blob.yaml
ts -G 1 -m -L 04_clora   python main.py --repeat=5 --configs configs/cifar100/04_clora.yaml
ts -G 1 -m -L 05_inflora python main.py --repeat=5 --configs configs/cifar100/05_inflora.yaml

# ImageNet-R
ts -G 1 -m -L 01_linear  python main.py --repeat=5 --configs configs/imagenetr/01_linear.yaml
ts -G 1 -m -L 02_lora    python main.py --repeat=5 --configs configs/imagenetr/02_lora.yaml
ts -G 1 -m -L 03_blob    python main.py --repeat=5 --configs configs/imagenetr/03_blob.yaml
ts -G 1 -m -L 04_clora   python main.py --repeat=5 --configs configs/imagenetr/04_clora.yaml
ts -G 1 -m -L 05_inflora python main.py --repeat=5 --configs configs/imagenetr/05_inflora.yaml

```
