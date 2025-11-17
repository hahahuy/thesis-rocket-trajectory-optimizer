"""
Training loop for residual network model.

Similar to train_pinn.py but uses ResidualNet and requires baseline trajectories.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.models import ResidualNet
from src.train.losses import PINNLoss
from src.train.callbacks import (
    EarlyStopping,
    CheckpointCallback,
    create_scheduler,
    LossWeightScheduler
)
from src.utils.loaders import create_dataloaders
from src.utils.reproducibility import set_seed

# Import train_epoch and validate from train_pinn
from src.train.train_pinn import train_epoch, validate, load_config, safe_float


def main():
    parser = argparse.ArgumentParser(description="Train Residual Network model")
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
    
    # Create experiment directory
    exp_name = train_cfg.get("experiment_name", "residual_baseline")
    exp_dir = Path(args.experiment_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load physics parameters (same as train_pinn.py)
    phys_path = config.get("physics_config", "configs/phys.yaml")
    scales_path = config.get("scales_config", "configs/scales.yaml")
    
    with open(phys_path, "r") as f:
        phys_config = yaml.safe_load(f)
    
    with open(scales_path, "r") as f:
        scales_config = yaml.safe_load(f)
    
    physics_params = {}
    if "aerodynamics" in phys_config:
        physics_params.update(phys_config["aerodynamics"])
    if "propulsion" in phys_config:
        physics_params.update(phys_config["propulsion"])
    if "atmosphere" in phys_config:
        physics_params.update(phys_config["atmosphere"])
    if "inertia" in phys_config:
        I_b = phys_config["inertia"]["I_b"]
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
    
    context_dim = train_loader.dataset.context_dim
    
    # Create model (ResidualNet instead of PINN)
    model = ResidualNet(
        context_dim=context_dim,
        n_hidden=int(model_cfg.get("n_hidden", 6)),
        n_neurons=int(model_cfg.get("n_neurons", 128)),
        activation=model_cfg.get("activation", "tanh"),
        fourier_features=int(model_cfg.get("fourier_features", 8)),
        layer_norm=bool(model_cfg.get("layer_norm", True)),
        dropout=safe_float(model_cfg.get("dropout"), 0.05),
        baseline_fn=None  # TODO: Implement baseline function from integrator
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function (same as PINN)
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
        checkpoint_dir=str(exp_dir / "checkpoints"),
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
    
    # Training loop (same structure as train_pinn.py)
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
        print(f"  Train Loss: {train_losses['total']:.6f}")
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
    with open(exp_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)
    
    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()

