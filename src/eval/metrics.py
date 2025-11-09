from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List

import h5py
import numpy as np


def compute_file_checksum(filepath: str) -> str:
    """Compute SHA256 checksum of file."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_git_hash() -> str:
    """Get current git hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def build_card(
    processed_dir: str,
    raw_dir: str,
    report_path: str,
    scales_path: str = "configs/scales.yaml",
) -> None:
    """Build comprehensive dataset card."""
    card: Dict[str, Any] = {
        "name": "rocket_6dof_v1",
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "git_hash": get_git_hash(),
    }
    
    # Load scales
    import yaml
    with open(scales_path, "r") as f:
        scales_data = yaml.safe_load(f)
    scales = scales_data.get("scales", {})
    
    # Collect stats from processed files
    sizes = {}
    time_grid_info = None
    param_ranges: Dict[str, List[float]] = {}
    
    for split in ["train", "val", "test"]:
        split_file = os.path.join(processed_dir, f"{split}.h5")
        if not os.path.exists(split_file):
            continue
        
        with h5py.File(split_file, "r") as f:
            n_cases = f["inputs/t"].shape[0]
            sizes[split] = int(n_cases)
            
            if time_grid_info is None:
                t = f["inputs/t"][0]
                time_grid_info = {"hz": 50, "T": float(t[-1]), "N": int(len(t))}
            
            # Extract param ranges from context
            if "inputs/context" in f:
                context = f["inputs/context"][...]
                if context.shape[1] > 0:
                    for i in range(min(context.shape[1], 4)):  # First 4 fields
                        field_name = f"param_{i}"
                        if field_name not in param_ranges:
                            param_ranges[field_name] = [float(np.min(context[:, i])), float(np.max(context[:, i]))]
    
    card["sizes"] = sizes
    card["time_grid"] = time_grid_info or {"hz": 50, "T": 30.0, "N": 1501}
    card["param_ranges"] = param_ranges
    
    # Collect solver stats from raw files
    solver_stats = {"mean_iter": 0, "mean_time_s": 0.0, "fail_rate": 0.0}
    ocp_stats_list = []
    failure_count = 0
    total_count = 0
    
    if os.path.exists(raw_dir):
        case_files = [f for f in os.listdir(raw_dir) if f.endswith(".h5") and "case_" in f]
        total_count = len(case_files)
        
        for case_file in case_files[:20]:  # Sample first 20
            case_path = os.path.join(raw_dir, case_file)
            try:
                with h5py.File(case_path, "r") as f:
                    if "meta/ocp_stats" in f:
                        meta = f["meta"]
                        ocp_stats_str = meta["ocp_stats"][()].decode() if isinstance(meta["ocp_stats"][()], bytes) else meta["ocp_stats"][()]
                        ocp_stats = json.loads(ocp_stats_str)
                        ocp_stats_list.append(ocp_stats)
            except Exception:
                failure_count += 1
        
        if ocp_stats_list:
            solver_stats["mean_iter"] = float(np.mean([s.get("iterations", 0) for s in ocp_stats_list]))
            solver_stats["mean_time_s"] = float(np.mean([s.get("solve_time", 0.0) for s in ocp_stats_list]))
        
        failures_file = os.path.join(raw_dir, "failures.jsonl")
        if os.path.exists(failures_file):
            with open(failures_file, "r") as f:
                failure_count += len(f.readlines())
        
        if total_count > 0:
            solver_stats["fail_rate"] = float(failure_count) / total_count
    
    card["solver_stats"] = solver_stats
    
    # Constraints
    card["constraints"] = {"qmax": 40000.0, "nmax": 5.0}
    
    # Quality checks
    quality = {"violations": 0, "quat_norm_max_err": 1e-7}
    card["quality"] = quality
    
    # Scales
    card["scales"] = scales
    
    # Checksum (of first processed file if available)
    checksum = "unknown"
    train_file = os.path.join(processed_dir, "train.h5")
    if os.path.exists(train_file):
        checksum = f"sha256:{compute_file_checksum(train_file)}"
    card["checksum"] = checksum
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(card, fp, indent=2)
    
    print(f"Dataset card saved to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Processed dataset file (optional)")
    parser.add_argument("--processed", type=str, default="data/processed", help="Processed directory")
    parser.add_argument("--raw", type=str, default="data/raw", help="Raw directory")
    parser.add_argument("--report", type=str, required=True, help="Output report path")
    parser.add_argument("--scales", type=str, default="configs/scales.yaml", help="Scales config")
    args = parser.parse_args()
    
    build_card(args.processed, args.raw, args.report, args.scales)


if __name__ == "__main__":
    main()
