#!/bin/bash -e
#SBATCH --job-name=domainnet_03_blob
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=log/domainnet_03_blob_%a.log
#SBATCH --array=0-4

export PATH=$NESI_PYVENV/bayescl/bin:$PATH

set -x # Echo commands to stdout
set -e # Exit on error

python main.py \
    --args study_name="run" num_workers=5 peft.save=True \
    --config configs/domainnet/03_blob.yaml \
    --seed "$SLURM_ARRAY_TASK_ID"