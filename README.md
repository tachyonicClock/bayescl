# bayescl

```bash
python main.py -c configs/${dataset}/${method}.yaml
```

## Hyperparameter Search

```bash
cd /home/antonlee/github.com/tachyonicClock/bayescl_frozen
conda activate cl
pip install -r requirements.txt
ts -G 1 -m -L 01_linear  python main.py --hpsearch --configs configs/cifar100/01_linear.yaml
ts -G 1 -m -L 02_lora    python main.py --hpsearch --configs configs/cifar100/02_lora.yaml
ts -G 1 -m -L 03_blob    python main.py --hpsearch --configs configs/cifar100/03_blob.yaml
ts -G 1 -m -L 04_clora   python main.py --hpsearch --configs configs/cifar100/04_clora.yaml
```

```
ts --set_gpu_free_perc=95
```
