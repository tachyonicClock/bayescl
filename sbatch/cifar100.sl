#!/bin/bash -e
#SBATCH --job-name=bayecl.CIFAR100
#SBATCH --time=00:02:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=log/runs/SplitCIFAR100/%x-%j.out
#SBATCH --error=log/runs/SplitCIFAR100/%x-%j.err
#SBATCH --array=0-0
# $SLURM_ARRAY_TASK_ID

set -x # Echo commands to stdout
set -e # Exit on error

# Override python version
export PATH=${HOME}/nobackup/pyvenv/bayescl/bin:${PATH}

run () {
    python main.py \
        --seed="$SLURM_ARRAY_TASK_ID" \
        --args study_name=run num_workers="$SLURM_CPUS_PER_TASK" \
        --config="$1"
}
run configs/cifar100/01_linear.yaml

    
