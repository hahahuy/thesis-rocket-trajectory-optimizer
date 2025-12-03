"""
Data preprocessing v2: Adds T_mag and q_dyn features.

This module extends the v1 preprocessing to compute:
1. T_mag(t): Thrust magnitude per timestep
2. q_dyn(t): Dynamic pressure per timestep

All v1 functionality is preserved. This is a new version that adds features.
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, Tuple

import h5py
import numpy as np

from .preprocess import (
    Scales,
    load_scales,
    to_nd,
    build_context_vector,
    CONTEXT_FIELDS,
)


def compute_thrust_magnitude(control: np.ndarray) -> np.ndarray:
    """
    Compute thrust magnitude from control array.
    
    Args:
        control: Control array [N, 4] where control[:, 0] is thrust magnitude T
        
    Returns:
        T_mag: Thrust magnitude [N] (already in control[:, 0])
    """
    # Control format: [T, theta_g, phi_g, delta]
    # T is already the magnitude
    return control[:, 0].copy()


def compute_dynamic_pressure(
    state: np.ndarray,
    scales: Scales,
    rho0: float = 1.225,
    h_scale: float = 8400.0
) -> np.ndarray:
    """
    Compute dynamic pressure q_dyn = 0.5 * rho * v^2.
    
    Args:
        state: State array [N, 14] (dimensional)
        scales: Scaling factors for nondimensionalization
        rho0: Sea level density [kg/m³]
        h_scale: Atmospheric scale height [m]
        
    Returns:
        q_dyn: Dynamic pressure [Pa] [N]
    """
    N = state.shape[0]
    
    # Extract velocity components (dimensional)
    vx = state[:, 3] * scales.V  # [m/s]
    vy = state[:, 4] * scales.V  # [m/s]
    vz = state[:, 5] * scales.V  # [m/s]
    
    # Compute speed
    speed = np.sqrt(vx**2 + vy**2 + vz**2)  # [m/s]
    
    # Extract altitude (dimensional)
    z = state[:, 2] * scales.L  # [m]
    
    # Compute atmospheric density: rho(z) = rho0 * exp(-z/H)
    z_clamped = np.clip(z, 0.0, None)  # Ensure non-negative
    rho = rho0 * np.exp(-z_clamped / h_scale)  # [kg/m³]
    
    # Compute dynamic pressure: q_dyn = 0.5 * rho * v^2
    q_dyn = 0.5 * rho * speed**2  # [Pa = N/m²]
    
    return q_dyn


def process_raw_to_splits_v2(
    raw_dir: str,
    processed_dir: str,
    scales_path: str,
    rho0: float = 1.225,
    h_scale: float = 8400.0
) -> None:
    """
    Process raw data to splits with v2 features (T_mag, q_dyn).
    
    This is identical to v1 preprocessing but adds:
    - inputs/T_mag: [n_cases, N] thrust magnitude per timestep
    - inputs/q_dyn: [n_cases, N] dynamic pressure per timestep
    
    Args:
        raw_dir: Directory containing raw HDF5 case files
        processed_dir: Output directory for processed splits
        scales_path: Path to scales.yaml
        rho0: Sea level density [kg/m³]
        h_scale: Atmospheric scale height [m]
    """
    os.makedirs(processed_dir, exist_ok=True)
    scales = load_scales(scales_path)

    # Discover cases
    cases = sorted([p for p in os.listdir(raw_dir) if p.endswith(".h5")])
    # Split by naming convention
    splits = {"train": [], "val": [], "test": []}
    for p in cases:
        if "case_train_" in p:
            splits["train"].append(p)
        elif "case_val_" in p:
            splits["val"].append(p)
        elif "case_test_" in p:
            splits["test"].append(p)

    # First pass: collect all context fields present across all cases
    all_context_keys = set()
    for split, files in splits.items():
        for fname in files:
            with h5py.File(os.path.join(raw_dir, fname), "r") as f:
                meta = f["meta"]
                params = json.loads(meta["params_used"][()].decode())
                all_context_keys.update(params.keys())
    
    # Build canonical context field list (only fields that exist)
    canonical_fields = [f for f in CONTEXT_FIELDS if f in all_context_keys]
    
    # Process each split
    for split, files in splits.items():
        if not files:
            continue
        xs, cs, ts, ctxs, t_mags, q_dyns = [], [], [], [], [], []
        
        for fname in files:
            with h5py.File(os.path.join(raw_dir, fname), "r") as f:
                t = f["time"][...]
                x = f["state"][...]  # Dimensional state
                u = f["control"][...]  # Dimensional control
                meta = f["meta"]
                # Extract context vector from params_used with normalization
                params = json.loads(meta["params_used"][()].decode())
                ctx_canonical = build_context_vector(params, scales, fields=canonical_fields)
            
            # Nondimensionalize state and control
            x_nd, u_nd, t_nd = to_nd(x, u, t, scales)
            
            # Compute T_mag from control (before nondimensionalization, T is already in [N])
            # After to_nd, u_nd[:, 0] is nondimensional, so we need to compute from original
            T_mag = compute_thrust_magnitude(u)  # [N] dimensional
            
            # Compute q_dyn from state (before nondimensionalization)
            q_dyn = compute_dynamic_pressure(x, scales, rho0=rho0, h_scale=h_scale)  # [Pa] dimensional
            
            # Nondimensionalize T_mag and q_dyn using scales
            # T_mag: [N] -> nondimensional using F scale
            T_mag_nd = T_mag / scales.F
            
            # q_dyn: [Pa = N/m²] -> nondimensional
            # q_dyn has units of pressure, which is F/L²
            # So we nondimensionalize as: q_dyn / (F / L²)
            q_dyn_nd = q_dyn / (scales.F / (scales.L**2))
            
            xs.append(x_nd)
            cs.append(u_nd)
            ts.append(t_nd)
            ctxs.append(ctx_canonical)
            t_mags.append(T_mag_nd)
            q_dyns.append(q_dyn_nd)
        
        # Stack and save
        with h5py.File(os.path.join(processed_dir, f"{split}.h5"), "w") as f:
            f.create_dataset("inputs/t", data=np.stack(ts), dtype="f8")
            f.create_dataset("inputs/context", data=np.stack(ctxs), dtype="f8")
            f.create_dataset("targets/state", data=np.stack(xs), dtype="f8")
            
            # V2 additions
            f.create_dataset("inputs/T_mag", data=np.stack(t_mags), dtype="f8")
            f.create_dataset("inputs/q_dyn", data=np.stack(q_dyns), dtype="f8")
            
            meta = f.create_group("meta")
            # NumPy 2.0 compatibility
            try:
                string_dtype = np.string_
            except AttributeError:
                string_dtype = np.bytes_
            scales_str = json.dumps(scales.__dict__).encode('utf-8')
            fields_str = json.dumps(canonical_fields).encode('utf-8')
            meta.create_dataset("scales", data=np.array(scales_str, dtype=string_dtype))
            meta.create_dataset("context_fields", data=np.array(fields_str, dtype=string_dtype))
            
            # Mark as v2 dataset
            meta.create_dataset("version", data=np.array("v2".encode('utf-8'), dtype=string_dtype))

    # Save split indices manifest
    with open(os.path.join(processed_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in splits.items()}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess raw rocket data with v2 features (T_mag, q_dyn)"
    )
    parser.add_argument("--raw", type=str, default="data/raw", help="Raw data directory")
    parser.add_argument("--out", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--scales", type=str, default="configs/scales.yaml", help="Scales config path")
    parser.add_argument("--rho0", type=float, default=1.225, help="Sea level density [kg/m³]")
    parser.add_argument("--h_scale", type=float, default=8400.0, help="Atmospheric scale height [m]")
    args = parser.parse_args()
    process_raw_to_splits_v2(args.raw, args.out, args.scales, rho0=args.rho0, h_scale=args.h_scale)


if __name__ == "__main__":
    main()

