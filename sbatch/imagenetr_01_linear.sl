#!/bin/bash -e
#SBATCH --job-name=imagenetr_01_linear
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=log/imagenetr_01_linear_%a.log
#SBATCH --array=0-0

export PATH=$NESI_PYVENV/bayescl/bin:$PATH

set -x # Echo commands to stdout
set -e # Exit on error

python main.py \
    --args study_name="run" num_workers=5 peft.save=True \
    --config configs/imagenetr/01_linear.yaml \
    --seed "$SLURM_ARRAY_TASK_ID"