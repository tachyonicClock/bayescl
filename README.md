# bayescl

```bash
python main.py -c configs/${dataset}/${method}.yaml
```

## Methodology
```
cuda9, cuda10, cuda16
```

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
rsync -a --info=progress2 lagerfield.ecs.vuw.ac.nz:$DATASETS/imagenet-r       $DATASETS
rsync -a --info=progress2 lagerfield.ecs.vuw.ac.nz:$DATASETS/domainnet        $DATASETS
```

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

```bash
ts -G 1 -m -L cifar_linea python main.py --hpsearch -c configs/cifar100/01_linear.yaml
ts -G 1 -m -L image_linea python main.py --hpsearch -c configs/imagenetr/01_linear.yaml
ts -G 1 -m -L domai_linea python main.py --hpsearch -c configs/domainnet/01_linear.yaml
ts -G 1 -m -L cifar_lora  python main.py --hpsearch -c configs/cifar100/02_lora.yaml
ts -G 1 -m -L image_lora  python main.py --hpsearch -c configs/imagenetr/02_lora.yaml
ts -G 1 -m -L domai_lora  python main.py --hpsearch -c configs/domainnet/02_lora.yaml
ts -G 1 -m -L cifar_blob  python main.py --hpsearch -c configs/cifar100/03_blob.yaml
ts -G 1 -m -L image_blob  python main.py --hpsearch -c configs/imagenetr/03_blob.yaml
ts -G 1 -m -L domai_blob  python main.py --hpsearch -c configs/domainnet/03_blob.yaml
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


```
sbatch sbatch/cifar100_01_linear.sl
sbatch sbatch/cifar100_02_lora.sl
sbatch sbatch/domainnet_01_linear.sl
sbatch sbatch/domainnet_02_lora.sl
sbatch sbatch/imagenetr_01_linear.sl
sbatch sbatch/imagenetr_02_lora.sl

sbatch sbatch/cifar100_03_blob.sl
sbatch sbatch/domainnet_03_blob.sl
sbatch sbatch/imagenetr_03_blob.sl

for f in sbatch/*.sl; do echo sbatch $f; done
```