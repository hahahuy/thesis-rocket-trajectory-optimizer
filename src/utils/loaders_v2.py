"""
Data loaders v2 for PINN training with T_mag and q_dyn features.

This module extends v1 loaders to include:
- T_mag: Thrust magnitude per timestep
- q_dyn: Dynamic pressure per timestep

All v1 functionality is preserved. This is a new version.
"""

import json
import os
from typing import Dict, Tuple, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from src.utils.loaders import CaseSampler  # Reuse v1 sampler


class RocketDatasetV2(Dataset):
    """
    Dataset v2 for rocket trajectory data with T_mag and q_dyn.
    
    Loads from processed HDF5 files with structure:
    - inputs/t: [n_cases, N] time grid
    - inputs/context: [n_cases, context_dim] context parameters
    - targets/state: [n_cases, N, 14] states
    - inputs/T_mag: [n_cases, N] thrust magnitude (v2)
    - inputs/q_dyn: [n_cases, N] dynamic pressure (v2)
    """
    
    def __init__(
        self,
        h5_path: str,
        max_cases: Optional[int] = None,
        time_subsample: Optional[int] = None
    ):
        """
        Args:
            h5_path: Path to processed HDF5 file (v2 format)
            max_cases: Maximum number of cases to load (None = all)
            time_subsample: Subsample time points (None = all, N = every Nth point)
        """
        self.h5_path = h5_path
        
        # Load metadata
        with h5py.File(h5_path, "r") as f:
            self.n_cases = f["inputs/t"].shape[0]
            self.N = f["inputs/t"].shape[1]
            self.context_dim = f["inputs/context"].shape[1]
            
            # Check if v2 features exist
            self.has_v2_features = "inputs/T_mag" in f and "inputs/q_dyn" in f
            
            # Load scales and context fields
            if "meta/scales" in f:
                scales_str = f["meta/scales"][()].decode() if isinstance(f["meta/scales"][()], bytes) else f["meta/scales"][()]
                self.scales = json.loads(scales_str)
            else:
                self.scales = {}
            
            if "meta/context_fields" in f:
                fields_str = f["meta/context_fields"][()].decode() if isinstance(f["meta/context_fields"][()], bytes) else f["meta/context_fields"][()]
                self.context_fields = json.loads(fields_str)
            else:
                self.context_fields = []
        
        self.max_cases = max_cases if max_cases is None else min(max_cases, self.n_cases)
        self.time_subsample = time_subsample
        
        # Pre-load data into memory (for faster access)
        with h5py.File(h5_path, "r") as f:
            self.t = f["inputs/t"][:self.max_cases]  # [n_cases, N]
            self.context = f["inputs/context"][:self.max_cases]  # [n_cases, context_dim]
            self.state = f["targets/state"][:self.max_cases]  # [n_cases, N, 14]
            
            # V2 features
            if self.has_v2_features:
                self.T_mag = f["inputs/T_mag"][:self.max_cases]  # [n_cases, N]
                self.q_dyn = f["inputs/q_dyn"][:self.max_cases]  # [n_cases, N]
            else:
                # Fallback: create zeros if v2 features missing (for backward compat)
                self.T_mag = np.zeros_like(self.t)
                self.q_dyn = np.zeros_like(self.t)
        
        # Subsample time if requested
        if time_subsample is not None:
            indices = np.arange(0, self.N, time_subsample)
            self.t = self.t[:, indices]
            self.state = self.state[:, indices]
            if self.has_v2_features:
                self.T_mag = self.T_mag[:, indices]
                self.q_dyn = self.q_dyn[:, indices]
            self.N = len(indices)
    
    def __len__(self) -> int:
        return self.max_cases if self.max_cases is not None else self.n_cases
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                - t: [N] time grid (nondimensional)
                - context: [context_dim] context vector (normalized)
                - state: [N, 14] state trajectory (nondimensional)
                - T_mag: [N] thrust magnitude (nondimensional, v2)
                - q_dyn: [N] dynamic pressure (nondimensional, v2)
                - case_id: int case index
        """
        return {
            "t": torch.tensor(self.t[idx], dtype=torch.float32),
            "context": torch.tensor(self.context[idx], dtype=torch.float32),
            "state": torch.tensor(self.state[idx], dtype=torch.float32),
            "T_mag": torch.tensor(self.T_mag[idx], dtype=torch.float32),
            "q_dyn": torch.tensor(self.q_dyn[idx], dtype=torch.float32),
            "case_id": idx
        }


def create_dataloaders_v2(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
    time_subsample: Optional[int] = None,
    max_train_cases: Optional[int] = None,
    max_val_cases: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders v2 (with T_mag and q_dyn).
    
    Args:
        data_dir: Directory containing processed splits (train.h5, val.h5, test.h5)
        batch_size: Batch size
        num_workers: Number of worker processes
        time_subsample: Subsample time points (None = all)
        max_train_cases: Maximum training cases (None = all)
        max_val_cases: Maximum validation cases (None = all)
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = RocketDatasetV2(
        os.path.join(data_dir, "train.h5"),
        max_cases=max_train_cases,
        time_subsample=time_subsample
    )
    
    val_dataset = RocketDatasetV2(
        os.path.join(data_dir, "val.h5"),
        max_cases=max_val_cases,
        time_subsample=time_subsample
    )
    
    test_dataset = RocketDatasetV2(
        os.path.join(data_dir, "test.h5"),
        max_cases=None,
        time_subsample=time_subsample
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=CaseSampler(train_dataset, shuffle=True),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader

