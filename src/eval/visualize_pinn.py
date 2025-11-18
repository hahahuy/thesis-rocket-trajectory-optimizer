"""
Visualization tools for PINN model evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    Plot predicted vs true trajectory with improved visualization.
    
    Args:
        t: Time [N] (nondimensional) or [batch, N] - will use first if batch
        pred_state: Predicted state [N, 14] or [batch, N, 14] (nondimensional)
        true_state: True state [N, 14] or [batch, N, 14] (nondimensional)
        scales: Scaling factors for dimensionalization
        save_path: Path to save figure (optional)
        title: Plot title
    """
    # Ensure we have single trajectories (not batches)
    if t.ndim > 1:
        t = t[0] if t.ndim == 2 else t.flatten()
    if pred_state.ndim > 2:
        pred_state = pred_state[0]
    if true_state.ndim > 2:
        true_state = true_state[0]
    
    # Validate shapes
    assert t.ndim == 1, f"Expected 1D time array, got shape {t.shape}"
    assert pred_state.ndim == 2 and pred_state.shape[1] == 14, f"Expected [N, 14] pred_state, got shape {pred_state.shape}"
    assert true_state.ndim == 2 and true_state.shape[1] == 14, f"Expected [N, 14] true_state, got shape {true_state.shape}"
    assert len(t) == pred_state.shape[0] == true_state.shape[0], "Time and state arrays must have same length"
    
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
    
    # Compute errors for better visualization
    error_altitude = (altitude_pred - altitude) / 1000  # km
    error_velocity = velocity_pred - velocity  # m/s
    error_mass = mass_pred - mass  # kg
    
    # Create figure with better layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Set style for better readability
    plt.style.use('default')
    
    # Color scheme: True = solid blue, Predicted = dashed red, Error = black
    true_color = '#1f77b4'  # Blue
    pred_color = '#d62728'  # Red
    error_color = '#2ca02c'  # Green for errors
    
    # 1. Altitude
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_dim, altitude / 1000, color=true_color, label='True', linewidth=2.5, alpha=0.9, zorder=2)
    ax1.plot(t_dim, altitude_pred / 1000, color=pred_color, label='Predicted', linewidth=2, 
             linestyle='--', alpha=0.9, zorder=2)
    ax1.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Altitude [km]', fontsize=11, fontweight='bold')
    ax1.set_title('Altitude', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax1.set_facecolor('#fafafa')
    
    # 2. Velocity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_dim, velocity, color=true_color, label='True', linewidth=2.5, alpha=0.9, zorder=2)
    ax2.plot(t_dim, velocity_pred, color=pred_color, label='Predicted', linewidth=2, 
             linestyle='--', alpha=0.9, zorder=2)
    ax2.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Velocity [m/s]', fontsize=11, fontweight='bold')
    ax2.set_title('Velocity Magnitude', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax2.set_facecolor('#fafafa')
    
    # 3. Mass
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t_dim, mass, color=true_color, label='True', linewidth=2.5, alpha=0.9, zorder=2)
    ax3.plot(t_dim, mass_pred, color=pred_color, label='Predicted', linewidth=2, 
             linestyle='--', alpha=0.9, zorder=2)
    ax3.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Mass [kg]', fontsize=11, fontweight='bold')
    ax3.set_title('Mass', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax3.set_facecolor('#fafafa')
    
    # 4. Dynamic Pressure
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t_dim, q_dyn / 1000, color=true_color, label='True', linewidth=2.5, alpha=0.9, zorder=2)
    ax4.plot(t_dim, q_dyn_pred / 1000, color=pred_color, label='Predicted', linewidth=2, 
             linestyle='--', alpha=0.9, zorder=2)
    ax4.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Dynamic Pressure [kPa]', fontsize=11, fontweight='bold')
    ax4.set_title('Dynamic Pressure', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=10, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax4.set_facecolor('#fafafa')
    
    # 5. Altitude Error
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t_dim, error_altitude, color=error_color, linewidth=2, alpha=0.8, zorder=2)
    ax5.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
    ax5.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Error [km]', fontsize=11, fontweight='bold')
    ax5.set_title('Altitude Error (Pred - True)', fontsize=12, fontweight='bold')
    rmse_alt = np.sqrt(np.mean(error_altitude**2))
    ax5.text(0.05, 0.95, f'RMSE: {rmse_alt:.4f} km', transform=ax5.transAxes, 
             fontsize=10, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax5.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax5.set_facecolor('#fafafa')
    
    # 6. Velocity Error
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t_dim, error_velocity, color=error_color, linewidth=2, alpha=0.8, zorder=2)
    ax6.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
    ax6.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Error [m/s]', fontsize=11, fontweight='bold')
    ax6.set_title('Velocity Error (Pred - True)', fontsize=12, fontweight='bold')
    rmse_vel = np.sqrt(np.mean(error_velocity**2))
    ax6.text(0.05, 0.95, f'RMSE: {rmse_vel:.4f} m/s', transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax6.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax6.set_facecolor('#fafafa')
    
    # 7. Position components (x, y, z)
    ax7 = fig.add_subplot(gs[2, 0])
    x_true = true_state[:, 0] * scales.L / 1000  # km
    y_true = true_state[:, 1] * scales.L / 1000  # km
    z_true = true_state[:, 2] * scales.L / 1000  # km
    x_pred = pred_state[:, 0] * scales.L / 1000
    y_pred = pred_state[:, 1] * scales.L / 1000
    z_pred = pred_state[:, 2] * scales.L / 1000
    ax7.plot(t_dim, x_true, color='#1f77b4', label='x (True)', linewidth=2, alpha=0.8)
    ax7.plot(t_dim, y_true, color='#ff7f0e', label='y (True)', linewidth=2, alpha=0.8)
    ax7.plot(t_dim, z_true, color='#2ca02c', label='z (True)', linewidth=2, alpha=0.8)
    ax7.plot(t_dim, x_pred, color='#d62728', label='x (Pred)', linewidth=1.5, 
             linestyle='--', alpha=0.6)
    ax7.plot(t_dim, y_pred, color='#9467bd', label='y (Pred)', linewidth=1.5, 
             linestyle='--', alpha=0.6)
    ax7.plot(t_dim, z_pred, color='#8c564b', label='z (Pred)', linewidth=1.5, 
             linestyle='--', alpha=0.6)
    ax7.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Position [km]', fontsize=11, fontweight='bold')
    ax7.set_title('Position Components', fontsize=12, fontweight='bold')
    ax7.legend(loc='best', fontsize=8, ncol=2, framealpha=0.9)
    ax7.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax7.set_facecolor('#fafafa')
    
    # 8. Mass Error
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(t_dim, error_mass, color=error_color, linewidth=2, alpha=0.8, zorder=2)
    ax8.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
    ax8.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Error [kg]', fontsize=11, fontweight='bold')
    ax8.set_title('Mass Error (Pred - True)', fontsize=12, fontweight='bold')
    rmse_mass = np.sqrt(np.mean(error_mass**2))
    ax8.text(0.05, 0.95, f'RMSE: {rmse_mass:.4f} kg', transform=ax8.transAxes, 
             fontsize=10, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax8.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax8.set_facecolor('#fafafa')
    
    # 9. Overall RMSE per component
    ax9 = fig.add_subplot(gs[2, 2])
    state_names = ["x", "y", "z", "vx", "vy", "vz", "q0", "q1", "q2", "q3", "wx", "wy", "wz", "m"]
    errors = pred_state - true_state
    rmse_per_comp = np.sqrt(np.mean(errors**2, axis=0))
    colors = plt.cm.viridis(np.linspace(0, 1, len(state_names)))
    bars = ax9.bar(range(len(state_names)), rmse_per_comp, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
    ax9.set_xlabel('State Component', fontsize=11, fontweight='bold')
    ax9.set_ylabel('RMSE (nondim)', fontsize=11, fontweight='bold')
    ax9.set_title('RMSE per State Component', fontsize=12, fontweight='bold')
    ax9.set_xticks(range(len(state_names)))
    ax9.set_xticklabels(state_names, rotation=45, ha='right', fontsize=9)
    ax9.grid(True, alpha=0.3, linestyle=':', linewidth=0.8, axis='y')
    ax9.set_yscale('log')
    ax9.set_facecolor('#fafafa')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    # Use subplots_adjust for better control and to avoid tight_layout warnings
    plt.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.05, hspace=0.35, wspace=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    else:
        plt.show()
    
    plt.close()


def plot_3d_trajectory(
    t: np.ndarray,
    pred_state: np.ndarray,
    true_state: np.ndarray,
    scales: Scales,
    save_path: Optional[str] = None,
    title: str = "3D Trajectory Comparison"
):
    """
    Plot 3D trajectory visualization comparing predicted vs true paths.
    
    Args:
        t: Time [N] (nondimensional) or [batch, N] - will use first if batch
        pred_state: Predicted state [N, 14] or [batch, N, 14] (nondimensional)
        true_state: True state [N, 14] or [batch, N, 14] (nondimensional)
        scales: Scaling factors for dimensionalization
        save_path: Path to save figure (optional)
        title: Plot title
    """
    # Ensure we have single trajectories (not batches)
    if t.ndim > 1:
        t = t[0] if t.ndim == 2 else t.flatten()
    if pred_state.ndim > 2:
        pred_state = pred_state[0]
    if true_state.ndim > 2:
        true_state = true_state[0]
    
    # Validate shapes
    assert t.ndim == 1, f"Expected 1D time array, got shape {t.shape}"
    assert pred_state.ndim == 2 and pred_state.shape[1] == 14, f"Expected [N, 14] pred_state, got shape {pred_state.shape}"
    assert true_state.ndim == 2 and true_state.shape[1] == 14, f"Expected [N, 14] true_state, got shape {true_state.shape}"
    assert len(t) == pred_state.shape[0] == true_state.shape[0], "Time and state arrays must have same length"
    
    # Extract position components and dimensionalize
    x_true = true_state[:, 0] * scales.L / 1000  # km
    y_true = true_state[:, 1] * scales.L / 1000  # km
    z_true = true_state[:, 2] * scales.L / 1000  # km (altitude)
    
    x_pred = pred_state[:, 0] * scales.L / 1000
    y_pred = pred_state[:, 1] * scales.L / 1000
    z_pred = pred_state[:, 2] * scales.L / 1000
    
    # Dimensionalize time
    t_dim, _, _ = from_nd(true_state, np.zeros((len(t), 4)), t, scales)
    
    # Create figure with 3D subplot
    fig = plt.figure(figsize=(16, 12))
    
    # Main 3D trajectory plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(x_true, y_true, z_true, 'b-', label='True', linewidth=2.5, alpha=0.9, zorder=2)
    ax1.plot(x_pred, y_pred, z_pred, 'r--', label='Predicted', linewidth=2, alpha=0.9, zorder=2)
    ax1.scatter(x_true[0], y_true[0], z_true[0], 
                color='green', s=150, marker='o', label='Start', zorder=5, 
                edgecolors='black', linewidths=2)
    ax1.scatter(x_true[-1], y_true[-1], z_true[-1], 
                color='red', s=150, marker='s', label='End', zorder=5,
                edgecolors='black', linewidths=2)
    ax1.set_xlabel('X [km]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y [km]', fontsize=12, fontweight='bold')
    ax1.set_zlabel('Altitude [km]', fontsize=12, fontweight='bold')
    ax1.set_title('3D Trajectory', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.view_init(elev=20, azim=45)
    ax1.grid(True, alpha=0.3)
    
    # 2D Ground track (X-Y projection)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(x_true, y_true, 'b-', label='True', linewidth=2.5, alpha=0.9, zorder=2)
    ax2.plot(x_pred, y_pred, 'r--', label='Predicted', linewidth=2, alpha=0.9, zorder=2)
    ax2.scatter(x_true[0], y_true[0], color='green', s=150, marker='o', 
                label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax2.scatter(x_true[-1], y_true[-1], color='red', s=150, marker='s', 
                label='End', zorder=5, edgecolors='black', linewidths=2)
    ax2.set_xlabel('X [km]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y [km]', fontsize=12, fontweight='bold')
    ax2.set_title('Ground Track (X-Y Projection)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10, framealpha=0.9)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax2.set_facecolor('#fafafa')
    
    # Altitude vs Horizontal Distance
    horizontal_dist_true = np.sqrt(x_true**2 + y_true**2)
    horizontal_dist_pred = np.sqrt(x_pred**2 + y_pred**2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(horizontal_dist_true, z_true, 'b-', label='True', linewidth=2.5, alpha=0.9, zorder=2)
    ax3.plot(horizontal_dist_pred, z_pred, 'r--', label='Predicted', linewidth=2, alpha=0.9, zorder=2)
    ax3.set_xlabel('Horizontal Distance [km]', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Altitude [km]', fontsize=12, fontweight='bold')
    ax3.set_title('Altitude vs Horizontal Distance', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax3.set_facecolor('#fafafa')
    
    # Trajectory error (3D distance)
    error_3d = np.sqrt((x_pred - x_true)**2 + (y_pred - y_true)**2 + (z_pred - z_true)**2)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(t_dim, error_3d, 'g-', linewidth=2, alpha=0.8, zorder=2)
    ax4.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax4.set_ylabel('3D Position Error [km]', fontsize=12, fontweight='bold')
    ax4.set_title('3D Trajectory Error', fontsize=14, fontweight='bold')
    rmse_3d = np.sqrt(np.mean(error_3d**2))
    ax4.text(0.05, 0.95, f'RMSE: {rmse_3d:.4f} km', transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax4.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax4.set_facecolor('#fafafa')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    # Use subplots_adjust for better control
    plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.08, hspace=0.35, wspace=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
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

