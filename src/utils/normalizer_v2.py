"""
Normalization utilities v2 for T_mag and q_dyn features.

Provides simple statistics-based normalization for the new v2 features.
This is optional - data is already nondimensionalized, but per-feature
normalization can help with training stability.
"""

import numpy as np
import torch
from typing import Dict, Optional


class ScalarNormalizer:
    """
    Simple normalizer for scalar features (T_mag, q_dyn).
    
    Stores mean and std statistics and provides normalize/denormalize methods.
    """
    
    def __init__(self):
        self.stats: Dict[str, Dict[str, float]] = {}
    
    def fit(self, T_mag: np.ndarray, q_dyn: np.ndarray, eps: float = 1e-6):
        """
        Fit normalization statistics from data.
        
        Args:
            T_mag: Thrust magnitude array [n_cases, N] or flattened
            q_dyn: Dynamic pressure array [n_cases, N] or flattened
            eps: Small epsilon to avoid division by zero
        """
        T_mag_flat = T_mag.flatten()
        q_dyn_flat = q_dyn.flatten()
        
        self.stats['T_mag'] = {
            'mean': float(np.mean(T_mag_flat)),
            'std': float(np.std(T_mag_flat)) + eps
        }
        
        self.stats['q_dyn'] = {
            'mean': float(np.mean(q_dyn_flat)),
            'std': float(np.std(q_dyn_flat)) + eps
        }
    
    def normalize_scalar(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """
        Normalize a scalar feature.
        
        Args:
            x: Input tensor [..., 1] or [..., N]
            key: Feature key ('T_mag' or 'q_dyn')
            
        Returns:
            Normalized tensor (same shape as input)
        """
        if key not in self.stats:
            # If stats not available, return as-is
            return x
        
        mean = self.stats[key]['mean']
        std = self.stats[key]['std']
        
        return (x - mean) / std
    
    def denormalize_scalar(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """
        Denormalize a scalar feature.
        
        Args:
            x: Normalized tensor [..., 1] or [..., N]
            key: Feature key ('T_mag' or 'q_dyn')
            
        Returns:
            Denormalized tensor (same shape as input)
        """
        if key not in self.stats:
            return x
        
        mean = self.stats[key]['mean']
        std = self.stats[key]['std']
        
        return x * std + mean
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current statistics."""
        return self.stats.copy()
    
    def set_stats(self, stats: Dict[str, Dict[str, float]]):
        """Set statistics (e.g., from saved config)."""
        self.stats = stats.copy()

