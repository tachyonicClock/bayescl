#!/bin/bash -e
#SBATCH --job-name=imagenetr_02_lora
#SBATCH --time=6:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=logs/imagenetr_02_lora_%a.log
#SBATCH --array=0

export PATH=$NESI_PYVENV/bayescl/bin:$PATH

set -x # Echo commands to stdout
set -e # Exit on error

python main.py --args label="run" --config configs/imagenetr/02_lora.yaml --seed "$SLURM_ARRAY_TASK_ID"
