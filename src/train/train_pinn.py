"""
Main training loop for PINN model.
"""

import argparse
import json
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


def train_epoch(
    model: nn.Module,
    train_loader,
    loss_fn: PINNLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    weight_scheduler: Optional[LossWeightScheduler] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {"data": 0.0, "physics": 0.0, "boundary": 0.0}
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
        state_pred = model(t, context)  # [batch, N, 14]
        
        # Compute loss
        loss, loss_dict = loss_fn(state_pred, state_true, t)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key].item()
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
    loss_components = {"data": 0.0, "physics": 0.0, "boundary": 0.0}
    n_batches = 0
    
    for batch in val_loader:
        t = batch["t"].to(device)
        context = batch["context"].to(device)
        state_true = batch["state"].to(device)
        
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        
        state_pred = model(t, context)
        loss, loss_dict = loss_fn(state_pred, state_true, t)
        
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key].item()
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
    model_type = model_cfg.get("type", "pinn")
    
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
    else:
        raise ValueError(
            "Unknown model type: "
            f"{model_type}. Supported: 'pinn', 'latent_ode', 'sequence', 'hybrid'"
        )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    loss_fn = PINNLoss(
        lambda_data=safe_float(loss_cfg.get("lambda_data"), 1.0),
        lambda_phys=safe_float(loss_cfg.get("lambda_phys"), 0.1),
        lambda_bc=safe_float(loss_cfg.get("lambda_bc"), 1.0),
        physics_params=physics_params,
        scales=scales
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
    
    # Create callbacks
    early_stopping = EarlyStopping(
        patience=int(train_cfg.get("early_stopping_patience", 10)),
        min_delta=safe_float(train_cfg.get("early_stopping_min_delta"), 0.0)
    )
    
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
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = checkpoint_callback.load(model, optimizer, args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    n_epochs = int(train_cfg.get("epochs", 100))
    best_val_loss = float('inf')
    train_log = []
    
    for epoch in range(start_epoch, n_epochs):
        # Train
        train_losses = train_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, weight_scheduler
        )
        
        # Validate
        val_losses = validate(model, val_loader, loss_fn, device)
        
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
            "lr": optimizer.param_groups[0]["lr"]
        }
        train_log.append(log_entry)
        
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {train_losses['total']:.6f} "
              f"(data: {train_losses['data']:.6f}, "
              f"phys: {train_losses['physics']:.6f}, "
              f"bc: {train_losses['boundary']:.6f})")
        print(f"  Val Loss: {val_losses['total']:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint
        is_best = val_losses["total"] < best_val_loss
        if is_best:
            best_val_loss = val_losses["total"]
        
        checkpoint_callback.save(
            model, optimizer, epoch, val_losses["total"], is_best
        )
        
        # Early stopping
        if early_stopping(val_losses["total"]):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save training log
    with open(logs_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)
    
    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()

