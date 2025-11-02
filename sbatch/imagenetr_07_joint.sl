#!/bin/bash -e
#SBATCH --job-name=imagenetr_07_joint
#SBATCH --time=4:36:06
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=log/imagenetr_07_joint_%a.log
#SBATCH --array=0-4

export PATH=$NESI_PYVENV/bayescl/bin:$PATH

set -x # Echo commands to stdout
set -e # Exit on error

python main.py \
    -c configs/imagenetr/07_joint.yaml \
    -a label.study="run" \
    -a num_workers=5 \
    -a seed="$SLURM_ARRAY_TASK_ID" \
    -a checkpoint=True \
    run