#!/bin/bash -e
#SBATCH --job-name=imagenetr_08_rwalk
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=log/imagenetr_08_rwalk_%a.log
#SBATCH --array=0-4

export PATH=$NESI_PYVENV/bayescl/bin:$PATH

set -x # Echo commands to stdout
set -e # Exit on error

python main.py \
    --args study_name="run" num_workers=5  \
    --config configs/imagenetr/08_rwalk.yaml \
    --seed "$SLURM_ARRAY_TASK_ID"