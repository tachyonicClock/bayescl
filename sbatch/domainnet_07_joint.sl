#!/bin/bash -e
#SBATCH --job-name=domainnet_07_joint
#SBATCH --time=6:36:30
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=log/domainnet_07_joint_%a.log
#SBATCH --array=0-4

export PATH=$NESI_PYVENV/bayescl/bin:$PATH

set -x # Echo commands to stdout
set -e # Exit on error

python main.py \
    -c configs/domainnet/07_joint.yaml \
    -a label.study="run" \
    -a num_workers=5 \
    -a seed="$SLURM_ARRAY_TASK_ID" \
    -a checkpoint=True \
    run