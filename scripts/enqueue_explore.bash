#!/bin/bash

set -e

scenario=${1:-'mnist'}
method=${2:-'00_naive'}
n_trials=5
config="configs/${scenario}/${method}.yaml"
n_gpu=$(nvidia-smi --list-gpus | wc -l)

# Run hyperparameter search to find the best hyperparameters for a new procedure.
# Usage: enqueue_adoption_hpsearch
enqueue_adoption_hpsearch () {
    ts -G 1 -L "hp ${scenario} ${method}" \
        notirun.sh adoption-hpsearch\
        python main.py -c "${config}" --epochs-scale=0.5 hpsearch
}

# Run many trials with the best hyperparameters found in the study (`--from-study`).
# This is used to make final decisions about which procedure to adopt.
# Usage: adoption_trials <id,...> (where id,... is a comma-separated list of ts IDs)
adoption_trials () {
    ts -G 1 -L "trial ${scenario} ${method}" -W "${1}" \
        notirun.sh adoption-trials\
        python main.py -c "${config}" --epochs-scale=0.5 \
            run --from-study --validate --n-trials="${n_trials}"
}

# Enqueue many jobs to fill all GPUs.
# Usage: enqueue_many <command>
# Returns a comma-separated list of ts IDs.
enqueue_many () {
    ts_task_ids=()
    for _ in $(seq 1 "${n_gpu}"); do
        ts_task_ids+=("$(eval "$1")")
    done
    printf '%s,' "${ts_task_ids[@]}"
}

# Ensure a clean git state. This is important for reproducibility.
if [ -n "$(git status --porcelain)" ]; then
    echo "Please ensure everything is committed."
    exit 1
fi

# Ensure DISCORD_WEBHOOK is set for notifications.
if [ -z "$DISCORD_WEBHOOK" ]; then
    echo "Please set DISCORD_WEBHOOK environment variable."
    exit 1
fi

# Ensure the config file exists.
if [ ! -f "$config" ]; then
    echo "Config file $config does not exist."
    exit 1
fi

echo "Enqueuing hyperparameter search jobs..."
hp_jobs=$(enqueue_many enqueue_adoption_hpsearch)
echo "Enqueuing adoption trials jobs..."
adoption_trials "$hp_jobs"
