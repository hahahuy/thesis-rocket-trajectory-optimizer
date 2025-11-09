from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import h5py
import numpy as np


@dataclass
class Scales:
    L: float
    V: float
    T: float
    M: float
    F: float
    W: float


def load_scales(path: str) -> Scales:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    s = y.get("scales", y)
    # Only include fields that are in Scales dataclass
    scale_fields = {"L", "V", "T", "M", "F", "W"}
    filtered = {k: v for k, v in s.items() if k in scale_fields}
    return Scales(**filtered)


def to_nd(state: np.ndarray, control: np.ndarray, t: np.ndarray, scales: Scales) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = state.copy()
    c = control.copy()
    tt = t.copy()
    # Nondimensionalization
    s[:, 0:3] /= scales.L
    s[:, 3:6] /= scales.V
    s[:, 10:13] /= scales.W
    s[:, 13:14] /= scales.M
    c[:, 0:1] /= scales.F
    tt[:] /= scales.T
    return s, c, tt


def from_nd(state: np.ndarray, control: np.ndarray, t: np.ndarray, scales: Scales) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = state.copy()
    c = control.copy()
    tt = t.copy()
    s[:, 0:3] *= scales.L
    s[:, 3:6] *= scales.V
    s[:, 10:13] *= scales.W
    s[:, 13:14] *= scales.M
    c[:, 0:1] *= scales.F
    tt[:] *= scales.T
    return s, c, tt


# Context vector field order (frozen)
CONTEXT_FIELDS = [
    "m0", "Isp", "Cd", "CL_alpha", "Cm_alpha", "S", "l_ref",
    "Tmax", "mdry", "gimbal_max_rad", "thrust_rate", "gimbal_rate_rad",
    "Ix", "Iy", "Iz", "rho0", "H",
    "wind_mag", "wind_dir_rad", "gust_amp", "gust_freq",
    "qmax", "nmax",
]


def build_context_vector(params: Dict[str, float], scales: Scales, fields: list = None) -> np.ndarray:
    """
    Build normalized context vector from params dict.
    
    Only includes fields present in params. Normalizes using physics-aware scaling.
    
    Args:
        params: Parameter dict
        scales: Scaling factors
        fields: List of fields to include (default: CONTEXT_FIELDS)
        
    Returns:
        Normalized context vector
    """
    if fields is None:
        fields = CONTEXT_FIELDS
    ctx_list = []
    for field in fields:
        if field not in params:
            ctx_list.append(0.0)  # Missing field -> zero
            continue
        val = params[field]
        # Physics-aware normalization
        if field in ["m0", "mdry"]:
            val_norm = val / scales.M
        elif field == "Tmax":
            val_norm = val / scales.F
        elif field in ["Ix", "Iy", "Iz"]:
            l_ref = params.get("l_ref", 1.2)
            val_norm = val / (scales.M * l_ref**2)
        elif field == "S":
            l_ref = params.get("l_ref", 1.2)
            val_norm = val / (l_ref**2)
        elif field in ["CL_alpha", "Cm_alpha", "Cd", "nmax"]:
            val_norm = val  # Already O(1)
        elif field == "Isp":
            val_norm = val / 250.0  # Reference Isp
        elif field == "rho0":
            val_norm = val / 1.225
        elif field == "H":
            val_norm = val / 8500.0
        elif field in ["wind_mag", "gust_amp"]:
            val_norm = val / scales.V
        elif field == "gust_freq":
            val_norm = val * scales.T  # 1/T scale
        else:
            # Angles and others: keep as-is (radians)
            val_norm = val
        ctx_list.append(val_norm)
    return np.array(ctx_list, dtype=float)


def process_raw_to_splits(raw_dir: str, processed_dir: str, scales_path: str) -> None:
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

    # Simple packer per split
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
    
    for split, files in splits.items():
        if not files:
            continue
        xs, cs, ts, ctxs = [], [], [], []
        for fname in files:
            with h5py.File(os.path.join(raw_dir, fname), "r") as f:
                t = f["time"][...]
                x = f["state"][...]
                u = f["control"][...]
                meta = f["meta"]
                # Extract context vector from params_used with normalization
                params = json.loads(meta["params_used"][()].decode())
                ctx_canonical = build_context_vector(params, scales, fields=canonical_fields)
            x_nd, u_nd, t_nd = to_nd(x, u, t, scales)
            xs.append(x_nd)
            cs.append(u_nd)
            ts.append(t_nd)
            ctxs.append(ctx_canonical)
        # Stack and save
        with h5py.File(os.path.join(processed_dir, f"{split}.h5"), "w") as f:
            f.create_dataset("inputs/t", data=np.stack(ts), dtype="f8")
            f.create_dataset("inputs/context", data=np.stack(ctxs), dtype="f8")
            f.create_dataset("targets/state", data=np.stack(xs), dtype="f8")
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

    # Save split indices manifest
    with open(os.path.join(processed_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in splits.items()}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default="data/processed")
    parser.add_argument("--scales", type=str, default="configs/scales.yaml")
    args = parser.parse_args()
    process_raw_to_splits(args.raw, args.out, args.scales)


if __name__ == "__main__":
    main()
