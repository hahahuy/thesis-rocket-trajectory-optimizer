"""
Training callbacks: learning rate schedulers, early stopping, checkpointing.
"""

import os
from typing import Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    ExponentialLR
)


class EarlyStopping:
    """
    Early stopping callback based on validation loss.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" (lower is better) or "max" (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self.mode == "min":
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == "max"
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


class CheckpointCallback:
    """
    Saves model checkpoints during training.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_best: bool = True,
        save_last: bool = True,
        save_frequency: int = 10
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best: Save best model based on validation loss
            save_last: Save last model
            save_frequency: Save checkpoint every N epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_last = save_last
        self.save_frequency = save_frequency
        self.best_loss = float('inf')
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        is_best: bool = False
    ):
        """
        Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            loss: Current loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        
        if is_best and self.save_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            self.best_loss = loss
        
        if self.save_last:
            torch.save(checkpoint, self.checkpoint_dir / 'last.pt')
        
        if epoch % self.save_frequency == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
    
    def load(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, path: str = None) -> int:
        """
        Load checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            path: Path to checkpoint (default: best.pt)
            
        Returns:
            Epoch number
        """
        if path is None:
            path = self.checkpoint_dir / 'best.pt'
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch']


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler ("cosine", "plateau", "step", "exponential")
        **kwargs: Scheduler-specific arguments
        
    Returns:
        Scheduler
    """
    if scheduler_type == "cosine":
        T_max = kwargs.get("T_max", 100)
        eta_min = kwargs.get("eta_min", 1e-6)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_type == "plateau":
        factor = kwargs.get("factor", 0.5)
        patience = kwargs.get("patience", 5)
        return ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    
    elif scheduler_type == "step":
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == "exponential":
        gamma = kwargs.get("gamma", 0.95)
        return ExponentialLR(optimizer, gamma=gamma)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class LossWeightScheduler:
    """
    Schedules loss weights during training (e.g., homotopy method).
    """
    
    def __init__(
        self,
        schedule_type: str = "linear",
        lambda_data_init: float = 1.0,
        lambda_data_final: float = 1.0,
        lambda_phys_init: float = 0.1,
        lambda_phys_final: float = 1.0,
        lambda_bc_init: float = 1.0,
        lambda_bc_final: float = 1.0,
        total_epochs: int = 100
    ):
        """
        Args:
            schedule_type: "linear", "exponential", or "fixed"
            lambda_*_init: Initial weight
            lambda_*_final: Final weight
            total_epochs: Total training epochs
        """
        self.schedule_type = schedule_type
        self.lambda_data_init = lambda_data_init
        self.lambda_data_final = lambda_data_final
        self.lambda_phys_init = lambda_phys_init
        self.lambda_phys_final = lambda_phys_final
        self.lambda_bc_init = lambda_bc_init
        self.lambda_bc_final = lambda_bc_final
        self.total_epochs = total_epochs
    
    def get_weights(self, epoch: int) -> Dict[str, float]:
        """
        Get loss weights for current epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
            
        Returns:
            Dict with lambda_data, lambda_phys, lambda_bc
        """
        if self.schedule_type == "fixed":
            return {
                "lambda_data": self.lambda_data_init,
                "lambda_phys": self.lambda_phys_init,
                "lambda_bc": self.lambda_bc_init
            }
        
        # Normalize epoch to [0, 1]
        t = min(epoch / self.total_epochs, 1.0)
        
        if self.schedule_type == "linear":
            lambda_data = self.lambda_data_init + t * (self.lambda_data_final - self.lambda_data_init)
            lambda_phys = self.lambda_phys_init + t * (self.lambda_phys_final - self.lambda_phys_init)
            lambda_bc = self.lambda_bc_init + t * (self.lambda_bc_final - self.lambda_bc_init)
        
        elif self.schedule_type == "exponential":
            lambda_data = self.lambda_data_init * (self.lambda_data_final / self.lambda_data_init) ** t
            lambda_phys = self.lambda_phys_init * (self.lambda_phys_final / self.lambda_phys_init) ** t
            lambda_bc = self.lambda_bc_init * (self.lambda_bc_final / self.lambda_bc_init) ** t
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return {
            "lambda_data": lambda_data,
            "lambda_phys": lambda_phys,
            "lambda_bc": lambda_bc
        }

