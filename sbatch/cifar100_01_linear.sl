#!/bin/bash -e
#SBATCH --job-name=cifar100_01_linear
#SBATCH --time=6:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=log/cifar100_01_linear_%a.log
#SBATCH --array=1-4

export PATH=$NESI_PYVENV/bayescl/bin:$PATH

set -x # Echo commands to stdout
set -e # Exit on error

python main.py --args study_name="run" --config configs/cifar100/01_linear.yaml --seed "$SLURM_ARRAY_TASK_ID"
