#!/bin/bash -e
#SBATCH --job-name=bayecl.cifar100
##SBATCH --time=06:00:00
#SBATCH --time=01:00:00
#SBATCH --mem=5G
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=log/runs/SplitCIFAR100/%x-%j.out
#SBATCH --error=log/runs/SplitCIFAR100/%x-%j.err
#SBATCH --array=0-4

set -x # Echo commands to stdout
set -e # Exit on error

module load Miniconda3

# Override python version
export PATH=${HOME}/nobackup/pyvenv/bayescl/bin:${PATH}

run () {
    python main.py \
        --seed="$SLURM_ARRAY_TASK_ID" \
        --args study_name=runs num_workers=4 \
        --config="$1"
}
# run configs/cifar100/01_linear.yaml
# run configs/cifar100/02_lora.yaml
# run configs/cifar100/03_blob.yaml
# run configs/cifar100/04_clora.yaml
run configs/cifar100/05_inflora.yaml
