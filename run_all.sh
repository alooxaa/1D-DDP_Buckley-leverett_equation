#!/usr/bin/env bash
# Runs the full experiment series.
#
# Six sweeps in order: first all three at 5000 Adam steps, then all three at
# 15000. Each step writes its own log into logs/. The script stops on the first
# failure so that later stages are not built from partial data.
#
# Usage:
#     ./run_all.sh                      # everything, roughly 9 hours
#     ./run_all.sh 5000                 # only the 5000-step sweeps
#     ./run_all.sh 15000                # only the 15000-step sweeps
#     ./run_all.sh report               # only recompute metrics and tables

set -euo pipefail
cd "$(dirname "$0")"

PY=../.venv/bin/python
TORCHRUN=../.venv/bin/torchrun
NPROC=2

mkdir -p logs sequential ddp_adam ddp_adam_lbfgs report

step() {          # step <log file> <title> <command...>
    local log="logs/$1"; shift
    local title="$1"; shift
    echo "=== $title | start $(date +%H:%M:%S) ==="
    if "$@" > "$log" 2>&1; then
        echo "=== $title | done $(date +%H:%M:%S) ==="
    else
        echo "!!! $title | FAILED, see $log" >&2
        tail -20 "$log" >&2
        exit 1
    fi
}

run_epoch() {     # run_epoch <number of Adam steps>
    local n="$1"

    step "sequential-$n.log" "1/3 sequential Adam+L-BFGS, $n steps" \
        $PY train_bl.py --adam-epochs "$n" --out "sequential/$n"

    step "ddp-adam-$n.log" "2/3 DDP Adam only, $n steps" \
        $TORCHRUN --standalone --nproc_per_node=$NPROC train_ddp.py \
            --adam-steps "$n" --out "ddp_adam/$n"

    step "ddp-adam-ref-$n.log" "2/3 reference 1 GPU full batch, $n steps" \
        $PY train_ddp.py --adam-steps "$n" --out "ddp_adam/$n" --baseline-reference

    step "ddp-adam-lbfgs-$n.log" "3/3 DDP Adam+L-BFGS, $n steps" \
        $TORCHRUN --standalone --nproc_per_node=$NPROC train_ddp_lbfgs.py \
            --adam-steps "$n" --out "ddp_adam_lbfgs/$n"
}

make_report() {
    step "eval_metrics.log" "metrics for every model" \
        env CUDA_VISIBLE_DEVICES="" $PY eval_metrics.py
    step "make_report.log" "tables and plots" \
        env CUDA_VISIBLE_DEVICES="" $PY make_report.py
}

echo "########## series started $(date +%F\ %H:%M:%S) ##########"

case "${1:-all}" in
    5000)   run_epoch 5000 ;;
    15000)  run_epoch 15000 ;;
    report) make_report ;;
    all)    run_epoch 5000; run_epoch 15000; make_report ;;
    *)      echo "unknown argument: $1" >&2; exit 2 ;;
esac

echo "########## series finished $(date +%F\ %H:%M:%S) ##########"
