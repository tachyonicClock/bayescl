#!/bin/bash -e
#SBATCH --job-name=cifar100_02_lora
#SBATCH --time=5:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=log/cifar100_02_lora_%a.log
#SBATCH --array=0-4

export PATH=$NESI_PYVENV/bayescl/bin:$PATH

set -x # Echo commands to stdout
set -e # Exit on error

python main.py \
    -c configs/cifar100/02_lora.yaml \
    -a study_name="run" \
    -a num_workers=5 \
    -a seed="$SLURM_ARRAY_TASK_ID" \
    run