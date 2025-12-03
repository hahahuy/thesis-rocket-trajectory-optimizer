#!/bin/bash
# Flexible evaluation wrapper that delegates to run_evaluation.py
#
# This script generates evaluation plots including:
# - Trajectory comparison plots (2D time series)
# - 3D trajectory plots with Parameters & Summary table
#   (showing initial parameters and trajectory statistics)
#
# The Parameters & Summary table is automatically included in 3D plots
# when context data is available from the dataset.

set -euo pipefail

CHECKPOINT=""
DATA_DIR="data/processed"
OUTPUT_DIR=""
CONFIG=""
DEVICE="auto"
BATCH_SIZE=8
NUM_WORKERS=0
TIME_SUBSAMPLE=""
MAX_PLOTS=3
NO_PLOTS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --time_subsample)
            TIME_SUBSAMPLE="$2"
            shift 2
            ;;
        --max_plots)
            MAX_PLOTS="$2"
            shift 2
            ;;
        --no_plots)
            NO_PLOTS=1
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "${CHECKPOINT}" ]]; then
    echo "Error: --checkpoint is required"
    exit 1
fi

CMD=(python run_evaluation.py
    --checkpoint "$CHECKPOINT"
    --data_dir "$DATA_DIR"
    --device "$DEVICE"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --max_plots "$MAX_PLOTS"
)

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD+=(--output_dir "$OUTPUT_DIR")
fi
if [[ -n "$CONFIG" ]]; then
    CMD+=(--config "$CONFIG")
fi
if [[ -n "$TIME_SUBSAMPLE" ]]; then
    CMD+=(--time_subsample "$TIME_SUBSAMPLE")
fi
if [[ "$NO_PLOTS" -eq 1 ]]; then
    CMD+=(--no_plots)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

