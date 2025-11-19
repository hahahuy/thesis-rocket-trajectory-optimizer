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
import yaml

from src.models import (
    PINN,
    RocketLatentODEPINN,
    RocketSequencePINN,
    RocketHybridPINN,
)
from src.utils.loaders import create_dataloaders
from src.eval.visualize_pinn import evaluate_model, plot_trajectory_comparison
from src.data.preprocess import load_scales


def safe_float(value, default=None):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def build_model(model_cfg, context_dim):
    model_type = model_cfg.get('type', 'pinn')

    if model_type == 'pinn':
        return PINN(
            context_dim=context_dim,
            n_hidden=int(model_cfg.get('n_hidden', 6)),
            n_neurons=int(model_cfg.get('n_neurons', 128)),
            activation=model_cfg.get('activation', 'tanh'),
            fourier_features=int(model_cfg.get('fourier_features', 8)),
            layer_norm=bool(model_cfg.get('layer_norm', True)),
            dropout=safe_float(model_cfg.get('dropout'), 0.05),
            context_embedding_dim=int(model_cfg.get('context_embedding_dim', 16)),
        )
    if model_type == 'latent_ode':
        return RocketLatentODEPINN(
            context_dim=context_dim,
            latent_dim=int(model_cfg.get('latent_dim', 64)),
            context_embedding_dim=int(model_cfg.get('context_embedding_dim', 64)),
            fourier_features=int(model_cfg.get('fourier_features', 8)),
            dynamics_n_hidden=int(model_cfg.get('dynamics_n_hidden', 3)),
            dynamics_n_neurons=int(model_cfg.get('dynamics_n_neurons', 128)),
            decoder_n_hidden=int(model_cfg.get('decoder_n_hidden', 3)),
            decoder_n_neurons=int(model_cfg.get('decoder_n_neurons', 128)),
            activation=model_cfg.get('activation', 'tanh'),
            layer_norm=bool(model_cfg.get('layer_norm', True)),
            dropout=safe_float(model_cfg.get('dropout'), 0.05),
        )
    if model_type == 'sequence':
        return RocketSequencePINN(
            context_dim=context_dim,
            context_embedding_dim=int(model_cfg.get('context_embedding_dim', 64)),
            fourier_features=int(model_cfg.get('fourier_features', 8)),
            d_model=int(model_cfg.get('d_model', 128)),
            n_layers=int(model_cfg.get('n_layers', 4)),
            n_heads=int(model_cfg.get('n_heads', 4)),
            dim_feedforward=int(model_cfg.get('dim_feedforward', 512)),
            dropout=safe_float(model_cfg.get('dropout'), 0.05),
            activation=model_cfg.get('transformer_activation', 'gelu'),
        )
    if model_type == 'hybrid':
        return RocketHybridPINN(
            context_dim=context_dim,
            latent_dim=int(model_cfg.get('latent_dim', 64)),
            context_embedding_dim=int(model_cfg.get('context_embedding_dim', 64)),
            fourier_features=int(model_cfg.get('fourier_features', 8)),
            d_model=int(model_cfg.get('d_model', 128)),
            n_layers=int(model_cfg.get('n_layers', 2)),
            n_heads=int(model_cfg.get('n_heads', 4)),
            dim_feedforward=int(model_cfg.get('dim_feedforward', 512)),
            encoder_window=int(model_cfg.get('encoder_window', 10)),
            activation=model_cfg.get('activation', 'tanh'),
            transformer_activation=model_cfg.get('transformer_activation', 'gelu'),
            dynamics_n_hidden=int(model_cfg.get('dynamics_n_hidden', 3)),
            dynamics_n_neurons=int(model_cfg.get('dynamics_n_neurons', 128)),
            decoder_n_hidden=int(model_cfg.get('decoder_n_hidden', 3)),
            decoder_n_neurons=int(model_cfg.get('decoder_n_neurons', 128)),
            layer_norm=bool(model_cfg.get('layer_norm', True)),
            dropout=safe_float(model_cfg.get('dropout'), 0.05),
        )

    raise ValueError(
        f\"Unknown model type: {model_type}. Supported: 'pinn', 'latent_ode', 'sequence', 'hybrid'\"
    )


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
_, _, test_loader = create_dataloaders(
    data_dir='$DATA_DIR',
    batch_size=8,
    num_workers=0
)

context_dim = test_loader.dataset.context_dim

# Create and load model
model = build_model(model_cfg, context_dim)
model.load_state_dict(model_state, strict=True)
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
        if i >= 3:
            break

        t = batch['t'].to(device)
        context = batch['context'].to(device)
        state_true = batch['state'].to(device)

        if t.dim() == 2:
            t = t.unsqueeze(-1)

        state_pred = model(t, context)
        
        # Convert to numpy (detach to remove gradient tracking)
        t_np = t[0].cpu().detach().squeeze(-1).numpy()
        pred_np = state_pred[0].cpu().detach().numpy()
        true_np = state_true[0].cpu().detach().numpy()
        
        plot_trajectory_comparison(
            t_np, pred_np, true_np, scales,
            save_path=str(output_dir / f'trajectory_case_{i}.png'),
            title=f'Case {i}'
        )

print('Evaluation complete!')
print(f'Evaluation results saved to: {output_dir}')
"

