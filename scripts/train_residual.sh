#!/bin/bash
# Training script for Residual Network model

set -e

# Default values
CONFIG_DIR="configs"
DATA_DIR="data/processed"
EXPERIMENT_DIR="experiments"
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --experiment_dir)
            EXPERIMENT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Merge configs if needed
if [ -z "$CONFIG" ]; then
    CONFIG="${CONFIG_DIR}/residual_config.yaml"
    python -c "
import yaml

with open('${CONFIG_DIR}/model.yaml', 'r') as f:
    model_cfg = yaml.safe_load(f)
with open('${CONFIG_DIR}/train.yaml', 'r') as f:
    train_cfg = yaml.safe_load(f)

# Set model type to residual
model_cfg['model']['type'] = 'residual'
train_cfg['train']['experiment_name'] = 'residual_baseline'

merged = {**model_cfg, **train_cfg}

with open('${CONFIG}', 'w') as f:
    yaml.dump(merged, f)
"
fi

# Run training
python -m src.train.train_residual \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --experiment_dir "$EXPERIMENT_DIR" \
    --seed "$SEED" \
    ${RESUME:+--resume "$RESUME"}

echo "Training complete!"

