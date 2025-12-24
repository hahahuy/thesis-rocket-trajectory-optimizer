#!/usr/bin/env python3
"""
Comprehensive evaluation script to generate all figures and tables for Section 3.

Generates:
- Section 3.1: Evaluation Setup (text info)
- Section 3.2: Trajectory Reconstruction Results (Figures 3.1-3.5)
- Section 3.3: Error Trajectories (Figures 3.6-3.8)
- Section 3.4: Aggregated Quantitative Metrics (Table 3.1, Figure 3.9)
- Section 3.5: Physics and Constraint Residuals (Figures 3.10-3.11)
- Section 3.6: Summary Table (Table 3.2)
"""

import os
import sys
import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

# Suppress OMP warning on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import PINN
from src.utils.loaders import create_dataloaders
from src.eval.visualize_pinn import evaluate_model
from src.data.preprocess import load_scales, Scales
from src.physics.dynamics_pytorch import compute_dynamics
from src.physics.physics_residual_layer import PhysicsResidualLayer
from run_evaluation import create_model, _forward_with_initial_state_if_needed, _requires_initial_state

# Set matplotlib style
plt.style.use('default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def compute_physics_residuals(
    t: np.ndarray,
    state_pred: np.ndarray,
    scales: Scales,
    physics_params: Dict,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute physics residuals for predicted trajectory using PhysicsResidualLayer.
    
    Returns:
        residual_norm: [N] norm of physics residual at each time step
        state_dot: [N, 14] time derivative of state
    """
    if device is None:
        device = torch.device("cpu")
    
    # Convert to torch tensors
    t_torch = torch.tensor(t, dtype=torch.float32, device=device)
    state_torch = torch.tensor(state_pred, dtype=torch.float32, device=device)
    
    # Ensure correct shape: [N, 14] and [N, 1] for time
    if state_torch.dim() == 1:
        state_torch = state_torch.unsqueeze(0)
    if t_torch.dim() == 0:
        t_torch = t_torch.unsqueeze(0)
    if t_torch.dim() == 1:
        t_torch = t_torch.unsqueeze(-1)  # [N, 1]
    
    # Create PhysicsResidualLayer
    scales_dict = {
        "L": scales.L,
        "V": scales.V,
        "T": scales.T,
        "M": scales.M,
        "F": scales.F,
        "W": scales.W,
    }
    
    residual_layer = PhysicsResidualLayer(
        physics_params=physics_params,
        scales=scales_dict
    ).to(device)
    residual_layer.eval()
    
    # Compute residuals
    with torch.no_grad():
        residuals = residual_layer(t_torch, state_torch)
        state_residual = residuals.state_residual  # [N, 14]
        state_dot = residuals.state_dot  # [N, 14]
    
    # Compute norm
    residual_norm = torch.linalg.norm(state_residual, dim=1).cpu().numpy()  # [N]
    state_dot_np = state_dot.cpu().numpy()  # [N, 14]
    
    return residual_norm, state_dot_np


def compute_mass_derivative(t: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Compute dm/dt using finite differences."""
    # Ensure 1D arrays and convert to float
    t = np.asarray(t, dtype=np.float64).flatten()
    mass = np.asarray(mass, dtype=np.float64).flatten()
    
    # Ensure same length
    N = min(len(t), len(mass))
    if N < 2:
        return np.zeros(N, dtype=np.float64)
    
    t = t[:N]
    mass = mass[:N]
    dm_dt = np.zeros(N, dtype=np.float64)
    
    for i in range(N):
        if i == 0:
            # Forward difference at first point
            if N > 1:
                dt = t[1] - t[0]
                if abs(dt) < 1e-10:
                    dt = 1e-3
                dm_dt[0] = (mass[1] - mass[0]) / dt
            else:
                dm_dt[0] = 0.0
        elif i == N - 1:
            # Backward difference at last point
            dt = t[N-1] - t[N-2]
            if abs(dt) < 1e-10:
                dt = 1e-3
            dm_dt[N-1] = (mass[N-1] - mass[N-2]) / dt
        else:
            # Central difference for interior points
            dt = t[i+1] - t[i-1]
            if abs(dt) < 1e-10:
                dt = 1e-3
            dm_dt[i] = (mass[i+1] - mass[i-1]) / (2.0 * dt)
    
    return dm_dt


def generate_figure_3_1(
    t: np.ndarray,
    pred_state: np.ndarray,
    true_state: np.ndarray,
    scales: Scales,
    output_path: Path
):
    """Figure 3.1: Vertical Position Trajectory (z vs time)"""
    t_flat = np.asarray(t, dtype=np.float64).flatten()
    t_dim = t_flat * scales.T  # Dimensionalize time directly
    z_ref = true_state[:, 2] * scales.L / 1000  # km
    z_pred = pred_state[:, 2] * scales.L / 1000  # km
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_dim, z_ref, 'b-', label='Reference (ODE solver)', linewidth=2.5, alpha=0.9)
    ax.plot(t_dim, z_pred, 'r--', label='PINN-predicted', linewidth=2, alpha=0.9)
    ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Vertical Position z [km]', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3.1: Vertical Position Trajectory', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_figure_3_2(
    t: np.ndarray,
    pred_state: np.ndarray,
    true_state: np.ndarray,
    scales: Scales,
    output_path: Path
):
    """Figure 3.2: Vertical Velocity Trajectory (vz vs time)"""
    t_flat = np.asarray(t, dtype=np.float64).flatten()
    t_dim = t_flat * scales.T  # Dimensionalize time directly
    vz_ref = true_state[:, 5] * scales.V  # m/s
    vz_pred = pred_state[:, 5] * scales.V  # m/s
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_dim, vz_ref, 'b-', label='Reference vz', linewidth=2.5, alpha=0.9)
    ax.plot(t_dim, vz_pred, 'r--', label='Predicted vz', linewidth=2, alpha=0.9)
    ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Vertical Velocity vz [m/s]', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3.2: Vertical Velocity Trajectory', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_figure_3_3(
    t: np.ndarray,
    pred_state: np.ndarray,
    true_state: np.ndarray,
    scales: Scales,
    output_path: Path
):
    """Figure 3.3: Mass Evolution (m vs time)"""
    t_flat = np.asarray(t, dtype=np.float64).flatten()
    t_dim = t_flat * scales.T  # Dimensionalize time directly
    m_ref = true_state[:, 13] * scales.M  # kg
    m_pred = pred_state[:, 13] * scales.M  # kg
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_dim, m_ref, 'b-', label='Reference mass', linewidth=2.5, alpha=0.9)
    ax.plot(t_dim, m_pred, 'r--', label='Predicted mass', linewidth=2, alpha=0.9)
    ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mass m [kg]', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3.3: Mass Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_figure_3_4(
    t: np.ndarray,
    pred_state: np.ndarray,
    true_state: np.ndarray,
    scales: Scales,
    output_path: Path
):
    """Figure 3.4: Horizontal Position Components (x, y vs time)"""
    t_flat = np.asarray(t, dtype=np.float64).flatten()
    t_dim = t_flat * scales.T  # Dimensionalize time directly
    x_ref = true_state[:, 0] * scales.L / 1000  # km
    y_ref = true_state[:, 1] * scales.L / 1000  # km
    x_pred = pred_state[:, 0] * scales.L / 1000  # km
    y_pred = pred_state[:, 1] * scales.L / 1000  # km
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    ax1.plot(t_dim, x_ref, 'b-', label='Reference x', linewidth=2.5, alpha=0.9)
    ax1.plot(t_dim, x_pred, 'r--', label='Predicted x', linewidth=2, alpha=0.9)
    ax1.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Horizontal Position x [km]', fontsize=12, fontweight='bold')
    ax1.set_title('Figure 3.4: Horizontal Position Components', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t_dim, y_ref, 'b-', label='Reference y', linewidth=2.5, alpha=0.9)
    ax2.plot(t_dim, y_pred, 'r--', label='Predicted y', linewidth=2, alpha=0.9)
    ax2.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Horizontal Position y [km]', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_figure_3_5(
    all_pred_states: List[np.ndarray],
    output_path: Path
):
    """Figure 3.5: Quaternion Norm Deviation over Time"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, pred_state in enumerate(all_pred_states[:10]):  # Show up to 10 trajectories
        quat = pred_state[:, 6:10]  # [N, 4]
        quat_norm = np.linalg.norm(quat, axis=1)
        quat_deviation = np.abs(quat_norm - 1.0)  # |∥q∥−1|
        t_norm = np.linspace(0, 30, len(quat_deviation))
        ax.plot(t_norm, quat_deviation, 'b-', linewidth=1, alpha=0.5 if i > 0 else 0.9)
    
    ax.axhline(0.0, color='r', linestyle='--', linewidth=2, label='Zero deviation (target)', alpha=0.7)
    ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Quaternion norm deviation |∥q∥−1|', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3.5: Quaternion Norm Deviation over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Use log scale for better visualization of small deviations
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_figure_3_6(
    t: np.ndarray,
    pred_state: np.ndarray,
    true_state: np.ndarray,
    scales: Scales,
    output_path: Path
):
    """Figure 3.6: Absolute Error in Vertical Position |z_pred − z_ref|"""
    t_flat = np.asarray(t, dtype=np.float64).flatten()
    t_dim = t_flat * scales.T  # Dimensionalize time directly
    z_ref = true_state[:, 2] * scales.L / 1000  # km
    z_pred = pred_state[:, 2] * scales.L / 1000  # km
    error = np.abs(z_pred - z_ref)
    
    # Find peak value and its location
    peak_idx = np.argmax(error)
    peak_time = t_dim[peak_idx]
    peak_value = error[peak_idx]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_dim, error, 'k-', linewidth=2, alpha=0.9)
    
    # Annotate peak value
    ax.plot(peak_time, peak_value, 'ro', markersize=10, zorder=5)
    ax.annotate(
        f'Peak: {peak_value:.4f} km\nat t = {peak_time:.2f} s',
        xy=(peak_time, peak_value),
        xytext=(peak_time + 2, peak_value * 1.1),
        fontsize=10,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=1.5),
        zorder=6
    )
    
    ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error |z_pred − z_ref| [km]', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3.6: Absolute Error in Vertical Position', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.8)  # Lighter grid
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_figure_3_7(
    t: np.ndarray,
    pred_state: np.ndarray,
    true_state: np.ndarray,
    scales: Scales,
    output_path: Path
):
    """Figure 3.7: Absolute Error in Vertical Velocity |vz_pred − vz_ref|"""
    t_flat = np.asarray(t, dtype=np.float64).flatten()
    t_dim = t_flat * scales.T  # Dimensionalize time directly
    vz_ref = true_state[:, 5] * scales.V  # m/s
    vz_pred = pred_state[:, 5] * scales.V  # m/s
    error = np.abs(vz_pred - vz_ref)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_dim, error, 'k-', linewidth=2, alpha=0.9)
    ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error |vz_pred − vz_ref| [m/s]', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3.7: Absolute Error in Vertical Velocity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_figure_3_8(
    t: np.ndarray,
    pred_state: np.ndarray,
    true_state: np.ndarray,
    scales: Scales,
    output_path: Path
):
    """Figure 3.8: Absolute Error in Mass |m_pred − m_ref|"""
    t_flat = np.asarray(t, dtype=np.float64).flatten()
    t_dim = t_flat * scales.T  # Dimensionalize time directly
    m_ref = true_state[:, 13] * scales.M  # kg
    m_pred = pred_state[:, 13] * scales.M  # kg
    error = np.abs(m_pred - m_ref)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_dim, error, 'k-', linewidth=2, alpha=0.9)
    ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error |m_pred − m_ref| [kg]', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3.8: Absolute Error in Mass', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_table_3_1(
    all_pred: np.ndarray,
    all_true: np.ndarray,
    output_path: Path
) -> pd.DataFrame:
    """Table 3.1: RMSE per State Variable (Test Set)"""
    state_names = [
        "x", "y", "z", "vx", "vy", "vz",
        "q0", "q1", "q2", "q3",
        "wx", "wy", "wz", "m"
    ]
    
    # Compute RMSE per trajectory per component
    # all_pred: [B, N, 14], all_true: [B, N, 14]
    errors = all_pred - all_true  # [B, N, 14]
    rmse_per_traj = np.sqrt(np.mean(errors**2, axis=1))  # [B, 14]
    
    # Compute mean and std across trajectories
    rmse_mean = np.mean(rmse_per_traj, axis=0)  # [14]
    rmse_std = np.std(rmse_per_traj, axis=0)  # [14]
    
    # Create DataFrame
    df = pd.DataFrame({
        'State Variable': state_names,
        'Mean RMSE': rmse_mean,
        'Standard Deviation': rmse_std,
        'Units': ['nondim'] * 14
    })
    
    # Save as CSV
    df.to_csv(output_path.with_suffix('.csv'), index=False)
    
    # Also save as LaTeX table
    latex_str = df.to_latex(index=False, float_format="%.6f")
    with open(output_path.with_suffix('.tex'), 'w') as f:
        f.write(latex_str)
    
    return df


def generate_figure_3_9(
    all_pred: np.ndarray,
    all_true: np.ndarray,
    output_path: Path
):
    """Figure 3.9: RMSE Distribution across Test Trajectories"""
    state_names = [
        "x", "y", "z", "vx", "vy", "vz",
        "q0", "q1", "q2", "q3",
        "wx", "wy", "wz", "m"
    ]
    
    # Compute RMSE per trajectory per component
    errors = all_pred - all_true  # [B, N, 14]
    rmse_per_traj = np.sqrt(np.mean(errors**2, axis=1))  # [B, 14]
    
    # Group by category
    position_indices = [0, 1, 2]  # x, y, z
    velocity_indices = [3, 4, 5]  # vx, vy, vz
    rotation_indices = [6, 7, 8, 9, 10, 11, 12]  # quaternion + angular rates
    mass_indices = [13]  # m
    
    position_rmse = rmse_per_traj[:, position_indices].flatten()
    velocity_rmse = rmse_per_traj[:, velocity_indices].flatten()
    rotation_rmse = rmse_per_traj[:, rotation_indices].flatten()
    mass_rmse = rmse_per_traj[:, mass_indices].flatten()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [position_rmse, velocity_rmse, rotation_rmse, mass_rmse]
    labels = ['Position', 'Velocity', 'Rotation', 'Mass']
    
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('RMSE (nondimensional)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3.6: RMSE Distribution across Test Trajectories', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_figure_3_10(
    t: np.ndarray,
    pred_state: np.ndarray,
    scales: Scales,
    physics_params: Dict,
    output_path: Path
):
    """Figure 3.10: Physics Residual Magnitudes over Time"""
    try:
        residual_norm, _ = compute_physics_residuals(
            t, pred_state, scales, physics_params
        )
        t_dim = t * scales.T
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(t_dim, residual_norm, 'k-', linewidth=2, alpha=0.9)
        ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Physics Residual Norm', fontsize=12, fontweight='bold')
        ax.set_title('Figure 3.7: Physics Residual Magnitudes over Time', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate Figure 3.10: {e}")
        # Create placeholder
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Physics residual computation failed:\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Figure 3.7: Physics Residual Magnitudes over Time', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def generate_figure_3_11(
    t: np.ndarray,
    pred_state: np.ndarray,
    true_state: np.ndarray,
    scales: Scales,
    output_path: Path
):
    """Figure 3.11: Mass Monotonicity Check"""
    # Ensure t is 1D
    t_flat = np.asarray(t, dtype=np.float64).flatten()
    
    # Dimensionalize time directly
    t_dim = t_flat * scales.T
    
    # Extract mass and ensure 1D
    m_pred = np.asarray(pred_state[:, 13] * scales.M, dtype=np.float64).flatten()  # kg
    m_ref = np.asarray(true_state[:, 13] * scales.M, dtype=np.float64).flatten()  # kg
    
    # Ensure same length
    min_len = min(len(t_dim), len(m_pred), len(m_ref))
    t_dim = t_dim[:min_len]
    m_pred = m_pred[:min_len]
    m_ref = m_ref[:min_len]
    
    dm_dt_pred = compute_mass_derivative(t_dim, m_pred)
    dm_dt_ref = compute_mass_derivative(t_dim, m_ref)
    
    # Count violations (dm/dt > 0)
    violations_pred = np.sum(dm_dt_pred > 0)
    violations_ref = np.sum(dm_dt_ref > 0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_dim, dm_dt_pred, 'r-', label=f'Predicted (violations: {violations_pred})', linewidth=2, alpha=0.9)
    ax.plot(t_dim, dm_dt_ref, 'b--', label=f'Reference (violations: {violations_ref})', linewidth=2, alpha=0.9)
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax.set_ylabel('dm/dt [kg/s]', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3.8: Mass Monotonicity Check', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_table_3_2(
    all_pred: np.ndarray,
    all_true: np.ndarray,
    all_residuals: Optional[List[np.ndarray]],
    test_set_size: int,
    output_path: Path
) -> pd.DataFrame:
    """Table 3.2: Summary of Key Observed Results"""
    
    # Compute metrics
    errors = all_pred - all_true  # [B, N, 14]
    
    # Vertical position accuracy
    z_errors = errors[:, :, 2]  # [B, N]
    z_rmse = np.sqrt(np.mean(z_errors**2))
    z_rmse_std = np.std(np.sqrt(np.mean(z_errors**2, axis=1)))
    
    # Velocity accuracy
    vz_errors = errors[:, :, 5]  # [B, N]
    vz_rmse = np.sqrt(np.mean(vz_errors**2))
    vz_rmse_std = np.std(np.sqrt(np.mean(vz_errors**2, axis=1)))
    
    # Mass behavior
    m_errors = errors[:, :, 13]  # [B, N]
    m_rmse = np.sqrt(np.mean(m_errors**2))
    m_rmse_std = np.std(np.sqrt(np.mean(m_errors**2, axis=1)))
    
    # Quaternion norm range
    quat_norms = []
    for pred in all_pred:
        quat = pred[:, 6:10]
        quat_norm = np.linalg.norm(quat, axis=1)
        quat_norms.extend(quat_norm.tolist())
    quat_norm_min = np.min(quat_norms)
    quat_norm_max = np.max(quat_norms)
    
    # Residual magnitude range
    if all_residuals:
        residual_mags = []
        for res in all_residuals:
            residual_mags.extend(res.tolist())
        residual_min = np.min(residual_mags)
        residual_max = np.max(residual_mags)
    else:
        residual_min = np.nan
        residual_max = np.nan
    
    df = pd.DataFrame({
        'Metric': [
            'Vertical position accuracy (RMSE)',
            'Velocity accuracy (RMSE)',
            'Mass behavior (RMSE)',
            'Quaternion norm range',
            'Residual magnitude range'
        ],
        'Observed Range': [
            f'{z_rmse:.6f} ± {z_rmse_std:.6f}',
            f'{vz_rmse:.6f} ± {vz_rmse_std:.6f}',
            f'{m_rmse:.6f} ± {m_rmse_std:.6f}',
            f'[{quat_norm_min:.6f}, {quat_norm_max:.6f}]',
            f'[{residual_min:.6f}, {residual_max:.6f}]' if not np.isnan(residual_min) else 'N/A'
        ],
        'Test Set Size': [test_set_size] * 5
    })
    
    # Save as CSV
    df.to_csv(output_path.with_suffix('.csv'), index=False)
    
    # Also save as LaTeX table
    latex_str = df.to_latex(index=False)
    with open(output_path.with_suffix('.tex'), 'w') as f:
        f.write(latex_str)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive evaluation report")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--data_dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for figures")
    parser.add_argument("--config", type=str, default=None, help="Config path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    
    # Setup
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint["model_state_dict"]
    
    experiment_dir = checkpoint_path.parent.parent
    logs_dir = experiment_dir / "logs"
    
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "evaluation_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load config
    config_candidates = []
    if args.config:
        config_candidates.append(Path(args.config))
    config_candidates.extend([
        logs_dir / "config.yaml",
        experiment_dir / "config.yaml",
        checkpoint_path.parent / "config.yaml",
    ])
    
    config = None
    for candidate in config_candidates:
        if candidate and candidate.exists():
            print(f"Loading config from {candidate}")
            with open(candidate, "r") as f:
                config = yaml.safe_load(f)
            break
    
    if config is None:
        raise FileNotFoundError("Config not found")
    
    # Load scales
    scales_path = config.get("scales_config", "configs/scales.yaml")
    scales = load_scales(scales_path)
    print(f"Scales loaded: L={scales.L}, V={scales.V}, T={scales.T}, M={scales.M}")
    
    # Load physics params
    physics_params = {}
    physics_config_path = config.get("physics_config", "configs/phys.yaml")
    if physics_config_path:
        physics_config_path = Path(physics_config_path)
        if not physics_config_path.is_file():
            physics_config_path = Path.cwd() / physics_config_path
        if physics_config_path.is_file():
            with open(physics_config_path, "r") as f:
                phys_config = yaml.safe_load(f) or {}
            for section in ("aerodynamics", "propulsion", "atmosphere"):
                physics_params.update(phys_config.get(section, {}))
            if "inertia" in phys_config and "I_b" in phys_config["inertia"]:
                I_b = phys_config["inertia"]["I_b"]
                if isinstance(I_b, list) and len(I_b) >= 9:
                    physics_params["I_b"] = [I_b[0], I_b[4], I_b[8]]
    
    physics_scales = {
        "L": scales.L, "V": scales.V, "T": scales.T,
        "M": scales.M, "F": scales.F, "W": scales.W,
    }
    
    # Load data
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Creating dataloaders from {data_dir}")
    train_loader, _, test_loader = create_dataloaders(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    context_dim = train_loader.dataset.context_dim
    test_set_size = len(test_loader.dataset)
    print(f"Test dataset size: {test_set_size}")
    
    # Create model
    model_cfg = config.get("model", {})
    model = create_model(model_cfg, context_dim, physics_params=physics_params, physics_scales=physics_scales)
    model.load_state_dict(model_state)
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    # Collect all predictions
    print("\n" + "=" * 60)
    print("Collecting predictions on test set...")
    print("=" * 60)
    
    all_pred = []
    all_true = []
    all_t = []
    all_pred_states_list = []
    all_residuals = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            t = batch["t"].to(device)
            context = batch["context"].to(device)
            state_true = batch["state"].to(device)
            
            if t.dim() == 2:
                t = t.unsqueeze(-1)
            
            state_pred = _forward_with_initial_state_if_needed(model, t, context, state_true)
            if isinstance(state_pred, (tuple, list)):
                state_pred = state_pred[0]
            
            all_pred.append(state_pred.cpu().numpy())
            all_true.append(state_true.cpu().numpy())
            all_t.append(t.cpu().numpy())
            
            # Store individual trajectories for figures
            for i in range(state_pred.shape[0]):
                t_np = t[i].cpu().detach().squeeze(-1).numpy()
                pred_np = state_pred[i].cpu().detach().numpy()
                all_pred_states_list.append(pred_np)
                
                # Try to compute residuals
                try:
                    residual_norm, _ = compute_physics_residuals(
                        t_np, pred_np, scales, physics_params, device
                    )
                    all_residuals.append(residual_norm)
                except:
                    pass
    
    all_pred = np.concatenate(all_pred, axis=0)  # [B, N, 14]
    all_true = np.concatenate(all_true, axis=0)  # [B, N, 14]
    all_t = np.concatenate(all_t, axis=0)  # [B, N] or [B, N, 1]
    
    # Select representative trajectory (first one)
    rep_idx = 0
    t_rep = all_t[rep_idx].flatten() if all_t[rep_idx].ndim > 1 else all_t[rep_idx]
    pred_rep = all_pred_states_list[rep_idx]
    true_rep = all_true[rep_idx]
    
    print(f"\nGenerating figures and tables...")
    print(f"Using trajectory {rep_idx} as representative case")
    
    # Generate Section 3.2: Trajectory Reconstruction Results
    print("Generating Section 3.2 figures...")
    try:
        generate_figure_3_1(t_rep, pred_rep, true_rep, scales, figures_dir / "figure_3_1_vertical_position.png")
        print("  [OK] Figure 3.1 generated")
    except Exception as e:
        print(f"  [ERROR] Error generating Figure 3.1: {e}")
        traceback.print_exc()
    
    try:
        generate_figure_3_2(t_rep, pred_rep, true_rep, scales, figures_dir / "figure_3_2_vertical_velocity.png")
        print("  [OK] Figure 3.2 generated")
    except Exception as e:
        print(f"  [ERROR] Error generating Figure 3.2: {e}")
        traceback.print_exc()
    
    try:
        generate_figure_3_3(t_rep, pred_rep, true_rep, scales, figures_dir / "figure_3_3_mass_evolution.png")
        print("  [OK] Figure 3.3 generated")
    except Exception as e:
        print(f"  [ERROR] Error generating Figure 3.3: {e}")
        traceback.print_exc()
    
    try:
        generate_figure_3_4(t_rep, pred_rep, true_rep, scales, figures_dir / "figure_3_4_horizontal_position.png")
        print("  [OK] Figure 3.4 generated")
    except Exception as e:
        print(f"  [ERROR] Error generating Figure 3.4: {e}")
        traceback.print_exc()
    
    try:
        generate_figure_3_5(all_pred_states_list, figures_dir / "figure_3_5_quaternion_norm.png")
        print("  [OK] Figure 3.5 generated")
    except Exception as e:
        print(f"  [ERROR] Error generating Figure 3.5: {e}")
        traceback.print_exc()
    
    # Generate Section 3.3: Error Trajectories
    print("Generating Section 3.3 figures...")
    try:
        generate_figure_3_6(t_rep, pred_rep, true_rep, scales, figures_dir / "figure_3_6_error_vertical_position.png")
        print("  [OK] Figure 3.6 generated")
    except Exception as e:
        print(f"  [ERROR] Error generating Figure 3.6: {e}")
        traceback.print_exc()
    
    try:
        generate_figure_3_7(t_rep, pred_rep, true_rep, scales, figures_dir / "figure_3_7_error_vertical_velocity.png")
        print("  [OK] Figure 3.7 generated")
    except Exception as e:
        print(f"  [ERROR] Error generating Figure 3.7: {e}")
        traceback.print_exc()
    
    try:
        generate_figure_3_8(t_rep, pred_rep, true_rep, scales, figures_dir / "figure_3_8_error_mass.png")
        print("  [OK] Figure 3.8 generated")
    except Exception as e:
        print(f"  [ERROR] Error generating Figure 3.8: {e}")
        traceback.print_exc()
    
    # Generate Section 3.4: Aggregated Metrics
    print("Generating Section 3.4 table and figure...")
    try:
        table_3_1 = generate_table_3_1(all_pred, all_true, tables_dir / "table_3_1_rmse_per_state")
        print("  [OK] Table 3.1 generated")
    except Exception as e:
        print(f"  [ERROR] Error generating Table 3.1: {e}")
        traceback.print_exc()
        table_3_1 = None
    
    try:
        generate_figure_3_9(all_pred, all_true, figures_dir / "figure_3_9_rmse_distribution.png")
        print("  [OK] Figure 3.9 generated")
    except Exception as e:
        print(f"  [ERROR] Error generating Figure 3.9: {e}")
        traceback.print_exc()
    
    # Generate Section 3.5: Physics Residuals
    print("Generating Section 3.5 figures...")
    try:
        generate_figure_3_10(t_rep, pred_rep, scales, physics_params, figures_dir / "figure_3_10_physics_residuals.png")
        print("  [OK] Figure 3.10 generated")
    except Exception as e:
        print(f"  [ERROR] Error generating Figure 3.10: {e}")
        traceback.print_exc()
    
    try:
        generate_figure_3_11(t_rep, pred_rep, true_rep, scales, figures_dir / "figure_3_11_mass_monotonicity.png")
        print("  [OK] Figure 3.11 generated")
    except Exception as e:
        print(f"  [ERROR] Error generating Figure 3.11: {e}")
        traceback.print_exc()
    
    # Generate Section 3.6: Summary Table
    print("Generating Section 3.6 table...")
    table_3_2 = generate_table_3_2(
        all_pred, all_true, all_residuals if all_residuals else None,
        test_set_size, tables_dir / "table_3_2_summary"
    )
    
    # Save evaluation setup info
    setup_info = {
        "test_set_size": test_set_size,
        "time_horizon": "0-30 s",
        "sampling_rate": 1501,
        "metrics": "RMSE, mean ± SD"
    }
    with open(output_dir / "evaluation_setup.json", "w") as f:
        json.dump(setup_info, f, indent=2)
    
    # Save metrics
    metrics = evaluate_model(model, test_loader, device, scales)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Evaluation report generation complete!")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Tables: {tables_dir}")
    print(f"  - Metrics: {output_dir / 'metrics.json'}")
    print(f"  - Setup info: {output_dir / 'evaluation_setup.json'}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

