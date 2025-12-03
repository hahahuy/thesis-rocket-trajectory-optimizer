"""
Main training loop for PINN model.
"""

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from src.models import PINN
from src.train.callbacks import (
    CheckpointCallback,
    EarlyStopping,
    LossWeightScheduler,
    create_scheduler
)
from src.train.losses import PINNLoss
from src.utils.loaders import create_dataloaders
from src.utils.reproducibility import set_seed


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def safe_float(value, default=None):
    """Safely convert value to float, handling strings like '1e-3'."""
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


def _requires_initial_state(model: nn.Module) -> bool:
    return bool(getattr(model, "requires_initial_state", False))


def _forward_with_initial_state_if_needed(
    model: nn.Module,
    t: torch.Tensor,
    context: torch.Tensor,
    state_true: torch.Tensor,
) -> torch.Tensor:
    if _requires_initial_state(model):
        initial_state = state_true[:, 0, :]
        return model(t, context, initial_state)
    return model(t, context)


def _sanitize_experiment_desc(description: str) -> str:
    """Normalize the experiment description for filesystem safety."""
    if not description:
        return "experiment"
    sanitized = re.sub(r"[^0-9a-zA-Z._-]+", "-", description.strip())
    sanitized = sanitized.strip("-_")
    return sanitized or "experiment"


def _next_experiment_index(root_dir: Path) -> int:
    """
    Return the next experiment index based on existing directories.
    
    Scans for directories matching pattern `expN_DD_MM_<desc>` and returns N+1.
    Handles edge cases:
    - Empty directory: returns 1
    - Non-matching directories: ignored
    - Invalid indices: skipped gracefully
    
    Note: Not thread-safe. Concurrent runs may create duplicate indices.
    For parallel experiments, use explicit experiment names or external coordination.
    
    Args:
        root_dir: Experiment root directory
        
    Returns:
        Next available experiment index (>= 1)
    """
    if not root_dir.exists():
        return 1
    
    pattern = re.compile(r"^exp(\d+)_\d{2}_\d{2}_.+")
    indices = []
    
    try:
        for child in root_dir.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match:
                try:
                    idx = int(match.group(1))
                    indices.append(idx)
                except (ValueError, IndexError):
                    # Skip invalid matches
                    continue
    except (PermissionError, OSError) as e:
        # If we can't read the directory, start from 1
        print(f"Warning: Could not read experiment directory: {e}")
        return 1
    
    return max(indices, default=0) + 1


class SoftLossScheduler:
    """
    Handles two-phase training for soft physics / smoothing losses (Direction D1.5).
    """

    def __init__(
        self,
        loss_fn: PINNLoss,
        schedule_cfg: Dict,
        total_epochs: int,
    ) -> None:
        self.loss_fn = loss_fn
        self.enabled = bool(schedule_cfg.get("enabled", False))
        self.total_epochs = max(1, int(total_epochs))
        if not self.enabled:
            self.keys = []
            return

        phase_ratio = safe_float(schedule_cfg.get("phase1_ratio"), 0.75)
        if phase_ratio is None:
            phase_ratio = 0.75
        phase_ratio = min(max(phase_ratio, 0.0), 1.0)
        self.phase_start = int(self.total_epochs * phase_ratio)
        self.ramp_type = schedule_cfg.get("ramp", "linear").lower()

        self.keys = [
            "lambda_mass_residual",
            "lambda_vz_residual",
            "lambda_vxy_residual",
            "lambda_smooth_z",
            "lambda_smooth_vz",
            "lambda_pos_vel",
            "lambda_smooth_pos",
            "lambda_zero_vxy",
            "lambda_zero_axy",
            "lambda_hacc",
            "lambda_xy_zero",
        ]
        self.targets = {
            key: float(getattr(self.loss_fn, key, 0.0)) for key in self.keys
        }
        # Initialize weights for phase 1
        initial_scale = 0.0 if self.phase_start > 0 else 1.0
        self._set_scale(initial_scale)

    def _set_scale(self, scale: float) -> None:
        if not self.enabled:
            return
        for key in self.keys:
            if hasattr(self.loss_fn, key):
                setattr(self.loss_fn, key, self.targets[key] * scale)

    def update(self, epoch: int) -> None:
        if not self.enabled:
            return
        if epoch < self.phase_start:
            self._set_scale(0.0)
            return
        # Handle short schedules (phase_start near total_epochs)
        remaining = max(1, self.total_epochs - self.phase_start)
        progress = min(max((epoch - self.phase_start) / remaining, 0.0), 1.0)
        if self.ramp_type == "cosine":
            scale = 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            scale = progress
        self._set_scale(scale)


def train_epoch(
    model: nn.Module,
    train_loader,
    loss_fn: PINNLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    weight_scheduler: Optional[LossWeightScheduler] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    # Initialize loss_components dynamically from loss_dict keys
    # We'll accumulate all keys that appear in loss_dict
    loss_components = {}
    n_batches = 0
    
    # Update loss weights if scheduler provided
    if weight_scheduler is not None:
        weights = weight_scheduler.get_weights(epoch)
        loss_fn.lambda_data = weights["lambda_data"]
        loss_fn.lambda_phys = weights["lambda_phys"]
        loss_fn.lambda_bc = weights["lambda_bc"]
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        t = batch["t"].to(device)  # [batch, N]
        context = batch["context"].to(device)  # [batch, context_dim]
        state_true = batch["state"].to(device)  # [batch, N, 14]
        
        # Ensure t has correct shape [batch, N, 1]
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        
        # Forward pass
        optimizer.zero_grad()
        model_out = _forward_with_initial_state_if_needed(model, t, context, state_true)

        # Some models (e.g. Direction AN) return (state_pred, physics_residuals).
        # For training we only need the state prediction here; residuals are
        # computed again inside the loss. This keeps older models fully intact.
        if isinstance(model_out, (tuple, list)):
            state_pred = model_out[0]
        else:
            state_pred = model_out

        # Compute loss
        loss, loss_dict = loss_fn(state_pred, state_true, t, context=context)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses - log ALL components from loss_dict
        total_loss += loss.item()
        for key, value in loss_dict.items():
            if key == "total":
                continue  # Skip total, we compute it separately
            if key not in loss_components:
                loss_components[key] = 0.0
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                loss_components[key] += value.item()
            else:
                loss_components[key] += float(value)
        n_batches += 1
    
    return {
        "total": total_loss / n_batches,
        **{k: v / n_batches for k, v in loss_components.items()}
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    loss_fn: PINNLoss,
    device: torch.device
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    # Initialize loss_components dynamically from loss_dict keys
    # We'll accumulate all keys that appear in loss_dict
    loss_components = {}
    n_batches = 0
    
    for batch in val_loader:
        t = batch["t"].to(device)
        context = batch["context"].to(device)
        state_true = batch["state"].to(device)
        
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        
        model_out = _forward_with_initial_state_if_needed(model, t, context, state_true)

        if isinstance(model_out, (tuple, list)):
            state_pred = model_out[0]
        else:
            state_pred = model_out

        loss, loss_dict = loss_fn(state_pred, state_true, t, context=context)

        total_loss += loss.item()
        # Accumulate losses - log ALL components from loss_dict
        for key, value in loss_dict.items():
            if key == "total":
                continue  # Skip total, we compute it separately
            if key not in loss_components:
                loss_components[key] = 0.0
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                loss_components[key] += value.item()
            else:
                loss_components[key] += float(value)
        n_batches += 1
    
    return {
        "total": total_loss / n_batches,
        **{k: v / n_batches for k, v in loss_components.items()}
    }


def main():
    parser = argparse.ArgumentParser(description="Train PINN model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--experiment_dir", type=str, default="experiments", help="Experiment directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    loss_cfg = config.get("loss", {})
    
    # Create experiment directory structure
    exp_root = Path(args.experiment_dir)
    exp_root.mkdir(parents=True, exist_ok=True)

    exp_desc = (
        train_cfg.get("experiment_desc")
        or train_cfg.get("experiment_name")
        or "pinn_baseline"
    )
    exp_desc = _sanitize_experiment_desc(exp_desc)
    date_stamp = datetime.now().strftime("%d_%m")
    exp_index = _next_experiment_index(exp_root)
    exp_dir = exp_root / f"exp{exp_index}_{date_stamp}_{exp_desc}"

    logs_dir = exp_dir / "logs"
    checkpoints_dir = exp_dir / "checkpoints"
    figures_dir = exp_dir / "figures"

    for directory in (logs_dir, checkpoints_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(logs_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Device - Force CPU for testing
    # [PINN_V2][2025-01-XX][Temporary] Force CPU for testing
    device = torch.device("cpu")
    print(f"Using device: {device} (forced CPU for testing)")
    
    # Load physics parameters
    phys_path = config.get("physics_config", "configs/phys.yaml")
    scales_path = config.get("scales_config", "configs/scales.yaml")
    
    with open(phys_path, "r") as f:
        phys_config = yaml.safe_load(f)
    
    with open(scales_path, "r") as f:
        scales_config = yaml.safe_load(f)
    
    # Extract physics params (nondimensionalize later if needed)
    physics_params = {}
    if "aerodynamics" in phys_config:
        physics_params.update(phys_config["aerodynamics"])
    if "propulsion" in phys_config:
        physics_params.update(phys_config["propulsion"])
    if "atmosphere" in phys_config:
        physics_params.update(phys_config["atmosphere"])
    if "inertia" in phys_config:
        I_b = phys_config["inertia"]["I_b"]
        # Extract diagonal
        physics_params["I_b"] = [I_b[0], I_b[4], I_b[8]]
    
    scales = scales_config.get("scales", {})
    
    # Create dataloaders
    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 0))
    time_subsample = train_cfg.get("time_subsample")
    if time_subsample is not None:
        time_subsample = int(time_subsample)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        time_subsample=time_subsample
    )
    
    # Get context dimension from dataset
    context_dim = train_loader.dataset.context_dim
    
    # [PINN_V2][2025-01-XX][Direction A]
    # Create model based on model_type configuration
    model_type = model_cfg.get("type", "pinn").lower()
    
    if model_type == "pinn":
        model = PINN(
            context_dim=context_dim,
            n_hidden=int(model_cfg.get("n_hidden", 6)),
            n_neurons=int(model_cfg.get("n_neurons", 128)),
            activation=model_cfg.get("activation", "tanh"),
            fourier_features=int(model_cfg.get("fourier_features", 8)),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.05)
        ).to(device)
    elif model_type == "direction_d":
        from src.models.direction_d_pinn import DirectionDPINN
        
        model = DirectionDPINN(
            context_dim=context_dim,
            fourier_features=int(model_cfg.get("fourier_features", 8)),
            context_embedding_dim=int(model_cfg.get("context_embedding_dim", 32)),
            backbone_hidden_dims=model_cfg.get("backbone_hidden_dims", [256, 256, 256, 256]),
            head_g3_hidden_dims=model_cfg.get("head_g3_hidden_dims", [128, 64]),
            head_g2_hidden_dims=model_cfg.get("head_g2_hidden_dims", [256, 128, 64]),
            head_g1_hidden_dims=model_cfg.get("head_g1_hidden_dims", [256, 128, 128, 64]),
            activation=model_cfg.get("activation", "gelu"),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.0),
        ).to(device)
    elif model_type == "direction_d1":
        from src.models.direction_d_pinn import DirectionDPINN_D1
        
        model = DirectionDPINN_D1(
            context_dim=context_dim,
            fourier_features=int(model_cfg.get("fourier_features", 8)),
            context_embedding_dim=int(model_cfg.get("context_embedding_dim", 32)),
            backbone_hidden_dims=model_cfg.get("backbone_hidden_dims", [256, 256, 256, 256]),
            head_g3_hidden_dims=model_cfg.get("head_g3_hidden_dims", [128, 64]),
            head_g2_hidden_dims=model_cfg.get("head_g2_hidden_dims", [256, 128, 64]),
            head_g1_hidden_dims=model_cfg.get("head_g1_hidden_dims", [256, 128, 128, 64]),
            activation=model_cfg.get("activation", "gelu"),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.0),
            integration_method=model_cfg.get("integration_method", "rk4"),
            use_physics_aware=bool(model_cfg.get("use_physics_aware", True)),
        ).to(device)
    elif model_type == "direction_d15":
        from src.models.direction_d_pinn import DirectionDPINN_D15

        model = DirectionDPINN_D15(
            context_dim=context_dim,
            fourier_features=int(model_cfg.get("fourier_features", 8)),
            context_embedding_dim=int(model_cfg.get("context_embedding_dim", 32)),
            backbone_hidden_dims=model_cfg.get("backbone_hidden_dims", [256, 256, 256, 256]),
            head_g3_hidden_dims=model_cfg.get("head_g3_hidden_dims", [128, 64]),
            head_g2_hidden_dims=model_cfg.get("head_g2_hidden_dims", [256, 128, 64]),
            head_g1_hidden_dims=model_cfg.get("head_g1_hidden_dims", [256, 128, 128, 64]),
            activation=model_cfg.get("activation", "gelu"),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.0),
            use_rotation_6d=bool(model_cfg.get("use_rotation_6d", True)),
            enforce_mass_monotonicity=bool(model_cfg.get("enforce_mass_monotonicity", False)),
        ).to(device)
    elif model_type == "direction_an":
        from src.models.direction_an_pinn import DirectionANPINN

        model = DirectionANPINN(
            context_dim=context_dim,
            fourier_features=int(model_cfg.get("fourier_features", 8)),
            stem_hidden_dim=int(model_cfg.get("stem_hidden_dim", 128)),
            stem_layers=int(model_cfg.get("stem_layers", 4)),
            activation=model_cfg.get("activation", "tanh"),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            translation_branch_dims=model_cfg.get("translation_branch_dims", [128, 128]),
            rotation_branch_dims=model_cfg.get("rotation_branch_dims", [256, 256]),
            mass_branch_dims=model_cfg.get("mass_branch_dims", [64]),
            dropout=safe_float(model_cfg.get("dropout"), 0.0),
            physics_params=physics_params,
            physics_scales=scales,
        ).to(device)
    elif model_type == "latent_ode":
        from src.models.latent_ode import RocketLatentODEPINN
        model = RocketLatentODEPINN(
            context_dim=context_dim,
            latent_dim=int(model_cfg.get("latent_dim", 64)),
            context_embedding_dim=int(model_cfg.get("context_embedding_dim", 64)),
            fourier_features=int(model_cfg.get("fourier_features", 8)),
            dynamics_n_hidden=int(model_cfg.get("dynamics_n_hidden", 3)),
            dynamics_n_neurons=int(model_cfg.get("dynamics_n_neurons", 128)),
            decoder_n_hidden=int(model_cfg.get("decoder_n_hidden", 3)),
            decoder_n_neurons=int(model_cfg.get("decoder_n_neurons", 128)),
            activation=model_cfg.get("activation", "tanh"),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.05)
        ).to(device)
    elif model_type == "sequence":
        from src.models.sequence_pinn import RocketSequencePINN

        model = RocketSequencePINN(
            context_dim=context_dim,
            context_embedding_dim=int(model_cfg.get("context_embedding_dim", 64)),
            fourier_features=int(model_cfg.get("fourier_features", 8)),
            d_model=int(model_cfg.get("d_model", 128)),
            n_layers=int(model_cfg.get("n_layers", 4)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
            dropout=safe_float(model_cfg.get("dropout"), 0.05),
            activation=model_cfg.get("transformer_activation", "gelu"),
        ).to(device)
    elif model_type == "hybrid":
        from src.models.hybrid_pinn import RocketHybridPINN

        model = RocketHybridPINN(
            context_dim=context_dim,
            latent_dim=int(model_cfg.get("latent_dim", 64)),
            context_embedding_dim=int(model_cfg.get("context_embedding_dim", 64)),
            fourier_features=int(model_cfg.get("fourier_features", 8)),
            d_model=int(model_cfg.get("d_model", 128)),
            n_layers=int(model_cfg.get("n_layers", 2)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
            encoder_window=int(model_cfg.get("encoder_window", 10)),
            activation=model_cfg.get("activation", "tanh"),
            transformer_activation=model_cfg.get("transformer_activation", "gelu"),
            dynamics_n_hidden=int(model_cfg.get("dynamics_n_hidden", 3)),
            dynamics_n_neurons=int(model_cfg.get("dynamics_n_neurons", 128)),
            decoder_n_hidden=int(model_cfg.get("decoder_n_hidden", 3)),
            decoder_n_neurons=int(model_cfg.get("decoder_n_neurons", 128)),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.05),
        ).to(device)
    elif model_type == "hybrid_c1":
        from src.models.hybrid_pinn import RocketHybridPINNC1

        model = RocketHybridPINNC1(
            context_dim=context_dim,
            latent_dim=int(model_cfg.get("latent_dim", 64)),
            context_embedding_dim=int(model_cfg.get("context_embedding_dim", 32)),
            fourier_features=int(model_cfg.get("fourier_features", 8)),
            d_model=int(model_cfg.get("d_model", 128)),
            n_layers=int(model_cfg.get("n_layers", 2)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
            encoder_window=int(model_cfg.get("encoder_window", 10)),
            activation=model_cfg.get("activation", "tanh"),
            transformer_activation=model_cfg.get("transformer_activation", "gelu"),
            dynamics_n_hidden=int(model_cfg.get("dynamics_n_hidden", 3)),
            dynamics_n_neurons=int(model_cfg.get("dynamics_n_neurons", 128)),
            decoder_n_hidden=int(model_cfg.get("decoder_n_hidden", 3)),
            decoder_n_neurons=int(model_cfg.get("decoder_n_neurons", 128)),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.05),
            debug_stats=bool(model_cfg.get("debug_stats", True)),
        ).to(device)
    elif model_type == "hybrid_c2":
        from src.models.hybrid_pinn import RocketHybridPINNC2

        model = RocketHybridPINNC2(
            context_dim=context_dim,
            latent_dim=int(model_cfg.get("latent_dim", 64)),
            fourier_features=int(model_cfg.get("fourier_features", 8)),
            shared_stem_hidden_dim=int(model_cfg.get("shared_stem_hidden_dim", 128)),
            temporal_type=model_cfg.get("temporal_type", "transformer"),
            temporal_n_layers=int(model_cfg.get("temporal_n_layers", 4)),
            temporal_n_heads=int(model_cfg.get("temporal_n_heads", 4)),
            temporal_dim_feedforward=int(model_cfg.get("temporal_dim_feedforward", 512)),
            encoder_window=int(model_cfg.get("encoder_window", 10)),
            translation_branch_dims=model_cfg.get("translation_branch_dims", [128, 128]),
            rotation_branch_dims=model_cfg.get("rotation_branch_dims", [256, 256]),
            mass_branch_dims=model_cfg.get("mass_branch_dims", [64]),
            activation=model_cfg.get("activation", "tanh"),
            transformer_activation=model_cfg.get("transformer_activation", "gelu"),
            dynamics_n_hidden=int(model_cfg.get("dynamics_n_hidden", 3)),
            dynamics_n_neurons=int(model_cfg.get("dynamics_n_neurons", 128)),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.05),
            debug_stats=bool(model_cfg.get("debug_stats", True)),
        ).to(device)
    elif model_type == "hybrid_c3":
        from src.models.hybrid_pinn import RocketHybridPINNC3

        model = RocketHybridPINNC3(
            context_dim=context_dim,
            latent_dim=int(model_cfg.get("latent_dim", 64)),
            fourier_features=int(model_cfg.get("fourier_features", 8)),
            shared_stem_hidden_dim=int(model_cfg.get("shared_stem_hidden_dim", 128)),
            temporal_type=model_cfg.get("temporal_type", "transformer"),
            temporal_n_layers=int(model_cfg.get("temporal_n_layers", 4)),
            temporal_n_heads=int(model_cfg.get("temporal_n_heads", 4)),
            temporal_dim_feedforward=int(model_cfg.get("temporal_dim_feedforward", 512)),
            encoder_window=int(model_cfg.get("encoder_window", 10)),
            translation_branch_dims=model_cfg.get("translation_branch_dims", [128, 128]),
            rotation_branch_dims=model_cfg.get("rotation_branch_dims", [256, 256]),
            mass_branch_dims=model_cfg.get("mass_branch_dims", [64]),
            activation=model_cfg.get("activation", "tanh"),
            transformer_activation=model_cfg.get("transformer_activation", "gelu"),
            dynamics_n_hidden=int(model_cfg.get("dynamics_n_hidden", 3)),
            dynamics_n_neurons=int(model_cfg.get("dynamics_n_neurons", 128)),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.05),
            debug_stats=bool(model_cfg.get("debug_stats", True)),
            use_physics_aware_translation=bool(model_cfg.get("use_physics_aware_translation", False)),
            use_coordinated_branches=bool(model_cfg.get("use_coordinated_branches", False)),
        ).to(device)
    else:
        raise ValueError(
            "Unknown model type: "
            f"{model_type}. Supported: 'pinn', 'latent_ode', 'sequence', 'hybrid', "
            "'hybrid_c1', 'hybrid_c2', 'hybrid_c3', 'direction_d', 'direction_d1', 'direction_d15'"
        )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function with enhanced parameters
    component_weights = loss_cfg.get("component_weights", None)
    loss_fn = PINNLoss(
        lambda_data=safe_float(loss_cfg.get("lambda_data"), 1.0),
        lambda_phys=safe_float(loss_cfg.get("lambda_phys"), 0.1),
        lambda_bc=safe_float(loss_cfg.get("lambda_bc"), 1.0),
        physics_params=physics_params,
        scales=scales,
        component_weights=component_weights,
        lambda_quat_norm=safe_float(loss_cfg.get("lambda_quat_norm"), 0.0),
        lambda_mass_flow=safe_float(loss_cfg.get("lambda_mass_flow"), 0.0),
        lambda_translation=safe_float(loss_cfg.get("lambda_translation"), 1.0),
        lambda_rotation=safe_float(loss_cfg.get("lambda_rotation"), 1.0),
        lambda_mass=safe_float(loss_cfg.get("lambda_mass"), 1.0),
        lambda_mass_residual=safe_float(loss_cfg.get("lambda_mass_residual"), 0.0),
        lambda_vz_residual=safe_float(loss_cfg.get("lambda_vz_residual"), 0.0),
        lambda_vxy_residual=safe_float(loss_cfg.get("lambda_vxy_residual"), 0.0),
        lambda_smooth_z=safe_float(loss_cfg.get("lambda_smooth_z"), 0.0),
        lambda_smooth_vz=safe_float(loss_cfg.get("lambda_smooth_vz"), 0.0),
        lambda_pos_vel=safe_float(loss_cfg.get("lambda_pos_vel"), 0.0),
        lambda_smooth_pos=safe_float(loss_cfg.get("lambda_smooth_pos"), 0.0),
        lambda_zero_vxy=safe_float(loss_cfg.get("lambda_zero_vxy"), 0.0),
        lambda_zero_axy=safe_float(loss_cfg.get("lambda_zero_axy"), 0.0),
        lambda_hacc=safe_float(loss_cfg.get("lambda_hacc"), 0.0),
        lambda_xy_zero=safe_float(loss_cfg.get("lambda_xy_zero"), 0.0),
    )
    
    # Create optimizer
    lr = safe_float(train_cfg.get("learning_rate"), 1e-3)
    weight_decay = safe_float(train_cfg.get("weight_decay"), 1e-5)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Create scheduler
    scheduler_cfg = train_cfg.get("scheduler", {})
    scheduler_kwargs = scheduler_cfg.get("kwargs", {})
    # Convert scheduler kwargs to proper types
    if "T_max" in scheduler_kwargs:
        scheduler_kwargs["T_max"] = int(scheduler_kwargs["T_max"])
    if "eta_min" in scheduler_kwargs:
        scheduler_kwargs["eta_min"] = safe_float(scheduler_kwargs["eta_min"], 1e-6)
    if "step_size" in scheduler_kwargs:
        scheduler_kwargs["step_size"] = int(scheduler_kwargs["step_size"])
    if "gamma" in scheduler_kwargs:
        scheduler_kwargs["gamma"] = safe_float(scheduler_kwargs["gamma"], 0.1)
    if "factor" in scheduler_kwargs:
        scheduler_kwargs["factor"] = safe_float(scheduler_kwargs["factor"], 0.5)
    if "patience" in scheduler_kwargs:
        scheduler_kwargs["patience"] = int(scheduler_kwargs["patience"])
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=scheduler_cfg.get("type", "cosine"),
        **scheduler_kwargs
    )
    
    # Create callbacks with phase-aware early stopping
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 10))
    early_stopping_patience_phase2 = int(train_cfg.get("early_stopping_patience_phase2", early_stopping_patience))
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=safe_float(train_cfg.get("early_stopping_min_delta"), 0.0)
    )
    # Phase 2 early stopping (will be created after phase_start is determined)
    early_stopping_phase2 = None
    
    checkpoint_callback = CheckpointCallback(
        checkpoint_dir=str(checkpoints_dir),
        save_best=True,
        save_last=True
    )
    
    # Loss weight scheduler
    weight_scheduler = None
    if loss_cfg.get("homotopy_schedule") != "fixed":
        weight_scheduler = LossWeightScheduler(
            schedule_type=loss_cfg.get("homotopy_schedule", "linear"),
            lambda_data_init=safe_float(loss_cfg.get("lambda_data"), 1.0),
            lambda_data_final=safe_float(loss_cfg.get("lambda_data"), 1.0),
            lambda_phys_init=safe_float(loss_cfg.get("lambda_phys"), 0.1),
            lambda_phys_final=safe_float(loss_cfg.get("lambda_phys_final"), 1.0),
            lambda_bc_init=safe_float(loss_cfg.get("lambda_bc"), 1.0),
            lambda_bc_final=safe_float(loss_cfg.get("lambda_bc"), 1.0),
            total_epochs=int(train_cfg.get("epochs", 100))
        )

    # Get number of epochs (needed for phase schedule setup)
    n_epochs = int(train_cfg.get("epochs", 100))
    
    phase_schedule_cfg = loss_cfg.get("phase_schedule")
    soft_loss_scheduler = None
    phase_start = None
    if phase_schedule_cfg and phase_schedule_cfg.get("enabled", False):
        phase1_ratio = safe_float(phase_schedule_cfg.get("phase1_ratio"), 0.75)
        phase_start = int(n_epochs * phase1_ratio)
        soft_loss_scheduler = SoftLossScheduler(
            loss_fn=loss_fn,
            schedule_cfg=phase_schedule_cfg,
            total_epochs=n_epochs,
        )
        # Create Phase 2 early stopping if needed
        if early_stopping_patience_phase2 > early_stopping_patience:
            early_stopping_phase2 = EarlyStopping(
                patience=early_stopping_patience_phase2,
                min_delta=safe_float(train_cfg.get("early_stopping_min_delta"), 0.0)
            )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = checkpoint_callback.load(model, optimizer, args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_val_loss = float('inf')
    best_val_metric = float('inf')  # Composite metric including D1.52 losses
    train_log = []
    
    def compute_composite_metric(val_losses: Dict[str, float]) -> float:
        """
        Compute composite validation metric that includes D1.52 losses.
        This metric is used for early stopping and best checkpoint selection.
        """
        # Base metric: total loss
        metric = val_losses.get("total", float('inf'))
        
        # Add D1.52 losses with weights (if present)
        d152_weight = 1.0  # Weight for D1.52 losses in composite metric
        d152_losses = [
            val_losses.get("zero_vxy", 0.0),
            val_losses.get("zero_axy", 0.0),
            val_losses.get("hacc", 0.0),
            val_losses.get("xy_zero", 0.0),
        ]
        d152_sum = sum(d152_losses)
        if d152_sum > 0:
            # Add D1.52 losses to metric (scaled by weight)
            metric += d152_weight * d152_sum
        
        return metric
    
    for epoch in range(start_epoch, n_epochs):
        if soft_loss_scheduler is not None:
            soft_loss_scheduler.update(epoch)
        # Train
        train_losses = train_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, weight_scheduler
        )
        
        # Validate
        val_losses = validate(model, val_loader, loss_fn, device)
        
        # Compute composite metric for early stopping and best checkpoint
        val_metric = compute_composite_metric(val_losses)
        
        # Update scheduler
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_losses["total"])
        else:
            scheduler.step()
        
        # Logging
        log_entry = {
            "epoch": epoch + 1,
            "train": train_losses,
            "val": val_losses,
            "val_metric": val_metric,  # Include composite metric in log
            "lr": optimizer.param_groups[0]["lr"]
        }
        train_log.append(log_entry)
        
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {train_losses['total']:.6f} "
              f"(data: {train_losses.get('data', 0):.6f}, "
              f"phys: {train_losses.get('physics', 0):.6f}, "
              f"bc: {train_losses.get('boundary', 0):.6f})")
        # Print D1.52 losses if present
        if 'zero_vxy' in train_losses:
            print(f"    D1.52: zero_vxy={train_losses.get('zero_vxy', 0):.6e}, "
                  f"zero_axy={train_losses.get('zero_axy', 0):.6e}, "
                  f"hacc={train_losses.get('hacc', 0):.6e}, "
                  f"xy_zero={train_losses.get('xy_zero', 0):.6e}")
        print(f"  Val Loss: {val_losses['total']:.6f}, Val Metric: {val_metric:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint using composite metric
        is_best = val_metric < best_val_metric
        if is_best:
            best_val_loss = val_losses["total"]
            best_val_metric = val_metric
        
        checkpoint_callback.save(
            model, optimizer, epoch, val_metric, is_best  # Use composite metric
        )
        
        # Early stopping - phase-aware
        in_phase2 = phase_start is not None and epoch >= phase_start
        if in_phase2 and early_stopping_phase2 is not None:
            # Use Phase 2 early stopping (higher patience)
            should_stop = early_stopping_phase2(val_metric)
        else:
            # Use Phase 1 early stopping
            should_stop = early_stopping(val_metric)
        
        if should_stop:
            phase_str = "Phase 2" if in_phase2 else "Phase 1"
            print(f"Early stopping at epoch {epoch+1} ({phase_str})")
            break
    
    # Save training log
    with open(logs_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)
    
    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()

