"""
Visualization tools for PINN model evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from src.models import PINN
from src.data.preprocess import from_nd, Scales


def plot_trajectory_comparison(
    t: np.ndarray,
    pred_state: np.ndarray,
    true_state: np.ndarray,
    scales: Scales,
    save_path: Optional[str] = None,
    title: str = "Trajectory Comparison"
):
    """
    Plot predicted vs true trajectory.
    
    Args:
        t: Time [N] (nondimensional)
        pred_state: Predicted state [N, 14] (nondimensional)
        true_state: True state [N, 14] (nondimensional)
        scales: Scaling factors for dimensionalization
        save_path: Path to save figure (optional)
        title: Plot title
    """
    # Dimensionalize
    t_dim, _, _ = from_nd(true_state, np.zeros((len(t), 4)), t, scales)
    
    # Extract key quantities
    altitude = true_state[:, 2] * scales.L  # z position
    altitude_pred = pred_state[:, 2] * scales.L
    
    velocity = np.linalg.norm(true_state[:, 3:6], axis=1) * scales.V
    velocity_pred = np.linalg.norm(pred_state[:, 3:6], axis=1) * scales.V
    
    mass = true_state[:, 13] * scales.M
    mass_pred = pred_state[:, 13] * scales.M
    
    # Dynamic pressure (approximate)
    rho = 1.225 * np.exp(-altitude / 8400.0)
    q_dyn = 0.5 * rho * velocity**2
    q_dyn_pred = 0.5 * rho * velocity_pred**2
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)
    
    # Altitude
    axes[0, 0].plot(t_dim, altitude / 1000, 'b-', label='True', linewidth=2)
    axes[0, 0].plot(t_dim, altitude_pred / 1000, 'r--', label='Predicted', linewidth=2)
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Altitude [km]')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_title('Altitude')
    
    # Velocity
    axes[0, 1].plot(t_dim, velocity, 'b-', label='True', linewidth=2)
    axes[0, 1].plot(t_dim, velocity_pred, 'r--', label='Predicted', linewidth=2)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Velocity [m/s]')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_title('Velocity')
    
    # Mass
    axes[1, 0].plot(t_dim, mass, 'b-', label='True', linewidth=2)
    axes[1, 0].plot(t_dim, mass_pred, 'r--', label='Predicted', linewidth=2)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Mass [kg]')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_title('Mass')
    
    # Dynamic pressure
    axes[1, 1].plot(t_dim, q_dyn / 1000, 'b-', label='True', linewidth=2)
    axes[1, 1].plot(t_dim, q_dyn_pred / 1000, 'r--', label='Predicted', linewidth=2)
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Dynamic Pressure [kPa]')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_title('Dynamic Pressure')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_residual_histogram(
    residuals: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Physics Residual Distribution"
):
    """
    Plot histogram of physics residuals.
    
    Args:
        residuals: Residual values [N] or [N, dim]
        save_path: Path to save figure (optional)
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if residuals.ndim == 2:
        residuals = residuals.flatten()
    
    ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Residual Value')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_res = np.mean(np.abs(residuals))
    max_res = np.max(np.abs(residuals))
    ax.axvline(mean_res, color='r', linestyle='--', label=f'Mean |res|: {mean_res:.2e}')
    ax.axvline(-mean_res, color='r', linestyle='--')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_loss_curves(
    train_log: list,
    save_path: Optional[str] = None
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_log: List of log entries from training
        save_path: Path to save figure (optional)
    """
    epochs = [entry["epoch"] for entry in train_log]
    train_loss = [entry["train"]["total"] for entry in train_log]
    val_loss = [entry["val"]["total"] for entry in train_log]
    
    train_data = [entry["train"]["data"] for entry in train_log]
    train_phys = [entry["train"]["physics"] for entry in train_log]
    train_bc = [entry["train"]["boundary"] for entry in train_log]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total loss
    axes[0].plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_yscale('log')
    
    # Component losses
    axes[1].plot(epochs, train_data, 'b-', label='Data', linewidth=2)
    axes[1].plot(epochs, train_phys, 'g-', label='Physics', linewidth=2)
    axes[1].plot(epochs, train_bc, 'm-', label='Boundary', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss Components')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    scales: Scales
) -> Dict[str, float]:
    """
    Evaluate model on test set and compute metrics.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
        scales: Scaling factors
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_pred = []
    all_true = []
    
    with torch.no_grad():
        for batch in test_loader:
            t = batch["t"].to(device)
            context = batch["context"].to(device)
            state_true = batch["state"].to(device)
            
            if t.dim() == 2:
                t = t.unsqueeze(-1)
            
            state_pred = model(t, context)
            
            all_pred.append(state_pred.cpu().numpy())
            all_true.append(state_true.cpu().numpy())
    
    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)
    
    # Compute RMSE per state component
    rmse = np.sqrt(np.mean((pred - true)**2, axis=(0, 1)))
    
    # Overall RMSE
    rmse_total = np.sqrt(np.mean((pred - true)**2))
    
    # State component names
    state_names = [
        "x", "y", "z", "vx", "vy", "vz",
        "q0", "q1", "q2", "q3",
        "wx", "wy", "wz", "m"
    ]
    
    metrics = {
        "rmse_total": float(rmse_total),
        "rmse_per_component": {name: float(rmse[i]) for i, name in enumerate(state_names)}
    }
    
    return metrics

