#!/usr/bin/env python3
"""
Quick visual checks for WP3 dataset health.

Generates plots for trajectory slices, parameter coverage, quaternion sanity, etc.
"""

import argparse
import json
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_slices(raw_dir: str, output_dir: str, n_cases: int = 3):
    """Plot trajectory slices for a few random cases."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find case files
    case_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".h5") and "case_" in f])
    if not case_files:
        print(f"Warning: No case files found in {raw_dir}")
        return
    
    n_plot = min(n_cases, len(case_files))
    selected = np.random.choice(len(case_files), n_plot, replace=False)
    
    fig, axes = plt.subplots(n_plot, 5, figsize=(20, 4 * n_plot))
    if n_plot == 1:
        axes = axes.reshape(1, -1)
    
    for idx, case_idx in enumerate(selected):
        case_file = os.path.join(raw_dir, case_files[case_idx])
        with h5py.File(case_file, "r") as f:
            t = f["time"][...]
            state = f["state"][...]
            control = f["control"][...]
            monitors = f["monitors"]
            q_dyn = monitors["q_dyn"][...]
            n_load = monitors["n_load"][...]
        
        # z(t)
        axes[idx, 0].plot(t, state[:, 2], "b-")
        axes[idx, 0].set_xlabel("t [s]")
        axes[idx, 0].set_ylabel("z [m]")
        axes[idx, 0].set_title(f"Altitude: {case_files[case_idx]}")
        axes[idx, 0].grid(True)
        
        # v_z(t) - vertical velocity (more relevant for ascent)
        axes[idx, 1].plot(t, state[:, 5], "g-")
        axes[idx, 1].set_xlabel("t [s]")
        axes[idx, 1].set_ylabel("v_z [m/s]")
        axes[idx, 1].set_title("Vertical Velocity")
        axes[idx, 1].grid(True)
        
        # mass(t)
        axes[idx, 2].plot(t, state[:, 13], "r-")
        axes[idx, 2].set_xlabel("t [s]")
        axes[idx, 2].set_ylabel("m [kg]")
        axes[idx, 2].set_title("Mass")
        axes[idx, 2].grid(True)
        
        # q_dyn(t)
        axes[idx, 3].plot(t, q_dyn / 1e4, "m-")
        axes[idx, 3].axhline(y=4.0, color="r", linestyle="--", label="qmax")
        axes[idx, 3].set_xlabel("t [s]")
        axes[idx, 3].set_ylabel("q_dyn [×10⁴ Pa]")
        axes[idx, 3].set_title("Dynamic Pressure")
        axes[idx, 3].legend()
        axes[idx, 3].grid(True)
        
        # n_load(t)
        axes[idx, 4].plot(t, n_load, "c-")
        axes[idx, 4].axhline(y=5.0, color="r", linestyle="--", label="nmax")
        axes[idx, 4].set_xlabel("t [s]")
        axes[idx, 4].set_ylabel("n_load [g]")
        axes[idx, 4].set_title("Load Factor")
        axes[idx, 4].legend()
        axes[idx, 4].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_slices.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/trajectory_slices.png")


def plot_parameter_coverage(processed_dir: str, raw_dir: str, output_dir: str):
    """Plot parameter coverage histograms and scatter plots using raw SI values."""
    os.makedirs(output_dir, exist_ok=True)
    
    splits = ["train", "val", "test"]
    params_data = {}
    
    # Extract raw parameter values from raw case files (SI, not normalized)
    for split in splits:
        split_file = os.path.join(processed_dir, f"{split}.h5")
        if not os.path.exists(split_file):
            continue
        
        # Get case files for this split
        case_files = sorted([f for f in os.listdir(raw_dir) if f.startswith(f"case_{split}_") and f.endswith(".h5")])
        if not case_files:
            continue
        
        m0_list, Isp_list, Cd_list, Tmax_list = [], [], [], []
        
        for case_file in case_files[:20]:  # Sample first 20
            case_path = os.path.join(raw_dir, case_file)
            try:
                with h5py.File(case_path, "r") as f:
                    if "meta/params_used" in f:
                        meta = f["meta"]
                        params_str = meta["params_used"][()].decode() if isinstance(meta["params_used"][()], bytes) else meta["params_used"][()]
                        params = json.loads(params_str)
                        m0_list.append(params.get("m0", 50.0))
                        Isp_list.append(params.get("Isp", 250.0))
                        Cd_list.append(params.get("Cd", 0.3))
                        Tmax_list.append(params.get("Tmax", 4000.0))
            except Exception:
                continue
        
        if m0_list:
            params_data[split] = {
                "m0": np.array(m0_list),
                "Isp": np.array(Isp_list),
                "Cd": np.array(Cd_list),
                "Tmax": np.array(Tmax_list),
            }
    
    if not params_data:
        print(f"Warning: No processed files found in {processed_dir}")
        return
    
    # Histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (param, label) in enumerate([("m0", "m0 [kg]"), ("Isp", "Isp [s]"), ("Cd", "Cd"), ("Tmax", "Tmax [N]")]):
        if idx >= len(axes):
            break
        ax = axes[idx]
        for split, data in params_data.items():
            if param in data and data[param] is not None:
                ax.hist(data[param], bins=20, alpha=0.5, label=split)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Parameter: {param}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_histograms.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/parameter_histograms.png")
    
    # Scatter: m0 vs Tmax
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for split, data in params_data.items():
        if "m0" in data and "Tmax" in data and data["m0"] is not None and data["Tmax"] is not None:
            ax.scatter(data["m0"], data["Tmax"], alpha=0.6, label=split, s=30)
    ax.set_xlabel("m0 [kg]")
    ax.set_ylabel("Tmax [N]")
    ax.set_title("Parameter Space Coverage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_scatter.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/parameter_scatter.png")


def plot_quaternion_sanity(raw_dir: str, output_dir: str, n_cases: int = 3):
    """Plot quaternion norm and attitude checks."""
    os.makedirs(output_dir, exist_ok=True)
    
    case_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".h5") and "case_" in f])
    if not case_files:
        return
    
    n_plot = min(n_cases, len(case_files))
    selected = np.random.choice(len(case_files), n_plot, replace=False)
    
    fig, axes = plt.subplots(n_plot, 2, figsize=(12, 4 * n_plot))
    if n_plot == 1:
        axes = axes.reshape(1, -1)
    
    for idx, case_idx in enumerate(selected):
        case_file = os.path.join(raw_dir, case_files[case_idx])
        with h5py.File(case_file, "r") as f:
            t = f["time"][...]
            state = f["state"][...]
            quat = state[:, 6:10]
        
        # Quaternion norm
        norms = np.linalg.norm(quat, axis=1)
        axes[idx, 0].plot(t, norms, "b-")
        axes[idx, 0].axhline(y=1.0, color="r", linestyle="--", label="unit")
        axes[idx, 0].set_xlabel("t [s]")
        axes[idx, 0].set_ylabel("||q||")
        axes[idx, 0].set_title(f"Quaternion Norm: {case_files[case_idx]}")
        axes[idx, 0].legend()
        axes[idx, 0].grid(True)
        
        # Quaternion components (to check for jumps)
        axes[idx, 1].plot(t, quat[:, 0], "r-", label="q_w", alpha=0.7)
        axes[idx, 1].plot(t, quat[:, 1], "g-", label="q_x", alpha=0.7)
        axes[idx, 1].plot(t, quat[:, 2], "b-", label="q_y", alpha=0.7)
        axes[idx, 1].plot(t, quat[:, 3], "m-", label="q_z", alpha=0.7)
        axes[idx, 1].set_xlabel("t [s]")
        axes[idx, 1].set_ylabel("Quaternion")
        axes[idx, 1].set_title("Quaternion Components")
        axes[idx, 1].legend()
        axes[idx, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quaternion_sanity.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/quaternion_sanity.png")


def plot_knots_vs_truth(raw_dir: str, output_dir: str, n_cases: int = 1):
    """Overlay OCP knots with integrated truth trajectory."""
    os.makedirs(output_dir, exist_ok=True)
    
    case_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".h5") and "case_" in f])
    if not case_files:
        return
    
    n_plot = min(n_cases, len(case_files))
    selected = np.random.choice(len(case_files), n_plot, replace=False)
    
    for idx, case_idx in enumerate(selected):
        case_file = os.path.join(raw_dir, case_files[case_idx])
        with h5py.File(case_file, "r") as f:
            t = f["time"][...]
            state = f["state"][...]
            ocp = f["ocp"]
            if "knots/state" in ocp and "t_knots" in ocp:
                x_knots = ocp["knots/state"][...]
                t_knots = ocp["t_knots"][...]
            else:
                print(f"Skipping {case_files[case_idx]}: no OCP knots")
                continue
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Plot z, v_z, mass, q_dyn
        for ax_idx, (state_idx, label) in enumerate([(2, "z [m]"), (5, "v_z [m/s]"), (13, "m [kg]")]):
            if ax_idx >= len(axes):
                break
            ax = axes[ax_idx]
            ax.plot(t, state[:, state_idx], "b-", label="Truth", alpha=0.7)
            ax.plot(t_knots, x_knots[:, state_idx], "ro", label="OCP knots", markersize=4)
            ax.set_xlabel("t [s]")
            ax.set_ylabel(label)
            ax.set_title(f"{label} - Truth vs OCP")
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"knots_vs_truth_{idx}.png"), dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/knots_vs_truth_{idx}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="data/raw")
    parser.add_argument("--processed", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="docs/figures/wp3_quick_checks")
    parser.add_argument("--n-cases", type=int, default=3)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("Generating quick check plots...")
    plot_trajectory_slices(args.raw, args.output, n_cases=args.n_cases)
    plot_parameter_coverage(args.processed, args.raw, args.output)
    plot_quaternion_sanity(args.raw, args.output, n_cases=args.n_cases)
    plot_knots_vs_truth(args.raw, args.output, n_cases=1)
    print(f"All plots saved to {args.output}")


if __name__ == "__main__":
    main()

