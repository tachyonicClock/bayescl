#!/bin/bash -e
#SBATCH --job-name=bayecl.CIFAR100
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=logs/runs/SplitCIFAR100/%x-%j.out
#SBATCH --error=logs/runs/SplitCIFAR100/%x-%j.out
#SBATCH --array=0-1
# $SLURM_ARRAY_TASK_ID

set -x # Echo commands to stdout
set -e # Exit on error

# Override python version
export PATH=${HOME}/nobackup/pyvenv/bayescl/bin:${PATH}
python main.py --config=configs/cifar100/01_linear.yaml --seed=$SLURM_ARRAY_TASK_ID
