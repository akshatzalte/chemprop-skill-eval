#!/bin/bash
# Run the ddp_train task once both GPUs are free, one run at a time.
#
# GPU 0 on the evaluation machine is shared with another user's long-running job; DDP needs both
# GPUs, so this waits until GPU 0 has been idle (under 5% utilisation and 1 GB memory) for ten
# minutes in a row before starting, and checks again before every run.
#
# usage: scripts/run_ddp_when_free.sh <results name> <model>...
set -u
NAME=$1
shift
cd "$(dirname "$0")"

gpu0_idle() {
    read -r util mem < <(nvidia-smi --id=0 --query-gpu=utilization.gpu,memory.used \
        --format=csv,noheader,nounits | tr -d ',')
    [ "$util" -lt 5 ] && [ "$mem" -lt 1024 ]
}

wait_for_idle() {
    local idle_minutes=0
    while [ "$idle_minutes" -lt 10 ]; do
        if gpu0_idle; then idle_minutes=$((idle_minutes + 1)); else idle_minutes=0; fi
        sleep 60
    done
}

for model in "$@"; do
    for run in 0 1 2; do
        for arm in without with with_short; do
            wait_for_idle
            echo "$(date -u +%H:%M) starting ddp_train $model $arm run $run"
            python3 run_eval.py --name "$NAME" --models "$model" --tasks ddp_train --arms "$arm" \
                --runs 1 --run-offset "$run" --budget-usd 10 --per-run-budget-usd 3 \
                --max-turns 60 --timeout-s 1800 --concurrency 1
        done
    done
done
