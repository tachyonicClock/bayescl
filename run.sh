#!/bin/bash
set -e

run_experiments () {
    set -e
    python main.py --configs configs/SplitCIFAR10/blob.yaml
    # python main.py --configs configs/SplitCIFAR10/blob.yaml --args peft.beta=100
    # python main.py --configs configs/SplitCIFAR10/blob.yaml --args peft.beta=1000

    # python main.py --configs configs/SplitCIFAR10/lora.yaml 
    # python main.py --configs configs/SplitCIFAR10/blob.yaml --args peft.beta=0 label="blob_beta0"
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

(run_experiments) &>> "$log_filename" || on_failure && on_success