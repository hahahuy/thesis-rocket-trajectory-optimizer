#!/bin/bash
# Evaluation script for PINN model

set -e

# Default values
CHECKPOINT=""
DATA_DIR="data/processed"
OUTPUT_DIR=""
CONFIG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint required"
    exit 1
fi

# Run evaluation
python -c "
import torch
import json
import numpy as np
from pathlib import Path
from src.models import PINN
from src.utils.loaders import create_dataloaders
from src.eval.visualize_pinn import evaluate_model, plot_trajectory_comparison
from src.data.preprocess import load_scales
import yaml

# Load checkpoint
checkpoint_path = Path('$CHECKPOINT').resolve()
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model_state = checkpoint['model_state_dict']

experiment_dir = checkpoint_path.parent.parent
logs_dir = experiment_dir / 'logs'

if '$OUTPUT_DIR':
    output_dir = Path('$OUTPUT_DIR')
else:
    output_dir = experiment_dir / 'figures'

output_dir.mkdir(parents=True, exist_ok=True)

# Load config
if '$CONFIG':
    with open('$CONFIG', 'r') as f:
        config = yaml.safe_load(f)
else:
    # Try to find config in checkpoint directory
    config = None
    for candidate in [
        logs_dir / 'config.yaml',
        experiment_dir / 'config.yaml',
        checkpoint_path.parent / 'config.yaml'
    ]:
        if candidate.exists():
            with open(candidate, 'r') as f:
                config = yaml.safe_load(f)
            break
    if config is None:
        raise FileNotFoundError('Config not found')

model_cfg = config.get('model', {})
scales_path = config.get('scales_config', 'configs/scales.yaml')

# Load scales
scales = load_scales(scales_path)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    data_dir='$DATA_DIR',
    batch_size=8,
    num_workers=0
)

context_dim = train_loader.dataset.context_dim

# Create model
model = PINN(
    context_dim=context_dim,
    n_hidden=model_cfg.get('n_hidden', 6),
    n_neurons=model_cfg.get('n_neurons', 128),
    activation=model_cfg.get('activation', 'tanh'),
    fourier_features=model_cfg.get('fourier_features', 8),
    layer_norm=model_cfg.get('layer_norm', True),
    dropout=model_cfg.get('dropout', 0.05)
)

model.load_state_dict(model_state)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Evaluate
print('Evaluating on test set...')
metrics = evaluate_model(model, test_loader, device, scales)

# Save metrics
metrics_path = output_dir / 'metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print('Metrics:')
print(json.dumps(metrics, indent=2))

# Plot sample trajectories
print('Generating sample plots...')
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        if i >= 3:  # Plot first 3 cases
            break
        
        t = batch['t'].to(device)
        context = batch['context'].to(device)
        state_true = batch['state'].to(device)
        
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        
        state_pred = model(t, context)
        
        # Convert to numpy
        t_np = t[0].cpu().squeeze(-1).numpy()
        pred_np = state_pred[0].cpu().numpy()
        true_np = state_true[0].cpu().numpy()
        
        plot_trajectory_comparison(
            t_np, pred_np, true_np, scales,
            save_path=str(output_dir / f'trajectory_case_{i}.png'),
            title=f'Case {i}'
        )

print('Evaluation complete!')
print(f'Evaluation results saved to: {output_dir}')
"

