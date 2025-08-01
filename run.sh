#!/bin/bash
set -e

run_experiments () {
    set -e
    set -x
    python main.py --hpsearch --configs configs/cifar100/00_base.yaml
    python main.py --hpsearch --configs configs/cifar100/01_linear.yaml
    python main.py --hpsearch --configs configs/cifar100/02_lora.yaml
    python main.py --hpsearch --configs configs/cifar100/03_blob.yaml
    python main.py --hpsearch --configs configs/cifar100/04_clora.yaml
}

log_filename=$(mktemp --suffix ".${USER}.log")
echo "Logging to $log_filename"

on_failure () {
    tail_filename=$(mktemp --suffix ".antonlee.tail.log")
    tail -n 100 "$log_filename" > "$tail_filename"
    discord.sh \
        --webhook-url=${DISCORD_WEBHOOK} \
        --text ":x: bayescl run failed" \
        --file "$tail_filename"
}

on_success () {
    discord.sh \
        --webhook-url=${DISCORD_WEBHOOK} \
        --text ":white_check_mark: bayescl run success"
}

if (run_experiments) &>> "$log_filename"; then
    on_success
else
    on_failure
fi