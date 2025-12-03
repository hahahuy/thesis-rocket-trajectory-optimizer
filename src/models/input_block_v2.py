"""
Input block v2: Fuses time, context, T_mag, and q_dyn features.

This module extends v1 input processing to include:
- T_mag: Thrust magnitude per timestep
- q_dyn: Dynamic pressure per timestep

All v1 functionality is preserved. This is a new version.
"""

import torch
import torch.nn as nn
from typing import Optional

from .architectures import FourierFeatures, ContextEncoder


class InputBlockV2(nn.Module):
    """
    Input fusion block v2 that combines:
    1. Time features (Fourier embedding)
    2. Context features (context encoder)
    3. T_mag and q_dyn (small MLP embedding)
    
    Output: Fused feature vector ready for model stem/backbone.
    """
    
    def __init__(
        self,
        context_dim: int,
        fourier_features: int = 8,
        context_embedding_dim: int = 32,
        extra_embedding_dim: int = 16,
        output_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        """
        Args:
            context_dim: Dimension of context vector
            fourier_features: Number of Fourier frequencies for time embedding
            context_embedding_dim: Output dimension of context encoder
            extra_embedding_dim: Output dimension for T_mag + q_dyn embedding
            output_dim: Optional projection dimension (if None, no projection)
            activation: Activation function for extra embedding MLP
            dropout: Dropout rate for extra embedding MLP
        """
        super().__init__()
        
        # Time embedding: t -> [t, sin(2πk t), cos(2πk t)] for k=1..K
        self.time_embed = FourierFeatures(n_frequencies=fourier_features)
        time_dim = 1 + 2 * fourier_features
        
        # Context embedding
        self.context_embed = ContextEncoder(
            context_dim=context_dim,
            embedding_dim=context_embedding_dim,
            activation=activation,
        )
        
        # Extra features embedding: T_mag + q_dyn -> embedding
        # Small MLP to embed the 2D [T_mag, q_dyn] vector
        layers = []
        layers.append(nn.Linear(2, extra_embedding_dim))
        if activation == "gelu":
            layers.append(nn.GELU())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "relu":
            layers.append(nn.ReLU())
        else:
            raise ValueError(f"Unknown activation: {activation}")
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.extra_embed = nn.Sequential(*layers)
        
        # Compute fused dimension
        fused_dim = time_dim + context_embedding_dim + extra_embedding_dim
        
        # Optional projection to output_dim
        if output_dim is not None:
            self.proj = nn.Linear(fused_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.proj = nn.Identity()
            self.output_dim = fused_dim
    
    def forward(
        self,
        t: torch.Tensor,
        context: torch.Tensor,
        T_mag: torch.Tensor,
        q_dyn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass fusing all input features.
        
        Args:
            t: Time [..., 1] or [batch, N, 1] (nondimensional)
            context: Context [..., context_dim] or [batch, context_dim]
            T_mag: Thrust magnitude [..., 1] or [batch, N, 1] (nondimensional)
            q_dyn: Dynamic pressure [..., 1] or [batch, N, 1] (nondimensional)
            
        Returns:
            fused: Fused features [..., output_dim] or [batch, N, output_dim]
        """
        # Ensure batched format
        if t.dim() == 2:
            t = t.unsqueeze(0)  # [1, N, 1]
        if context.dim() == 1:
            context = context.unsqueeze(0)  # [1, context_dim]
        if T_mag.dim() == 1:
            T_mag = T_mag.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        if q_dyn.dim() == 1:
            q_dyn = q_dyn.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        
        batch_size, N, _ = t.shape
        
        # 1. Time features
        t_features = self.time_embed(t)  # [batch, N, time_dim]
        
        # 2. Context features (broadcast to match time dimension)
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(batch_size, N, -1)  # [batch, N, context_dim]
        elif context.dim() == 3 and context.shape[1] == 1:
            context = context.expand(batch_size, N, -1)
        
        c_features = self.context_embed(context)  # [batch, N, context_embedding_dim]
        
        # 3. Extra features: T_mag + q_dyn
        # Ensure T_mag and q_dyn have correct shape
        if T_mag.dim() == 2:
            T_mag = T_mag.unsqueeze(-1)  # [batch, N, 1]
        if q_dyn.dim() == 2:
            q_dyn = q_dyn.unsqueeze(-1)  # [batch, N, 1]
        
        extra = torch.cat([T_mag, q_dyn], dim=-1)  # [batch, N, 2]
        extra_features = self.extra_embed(extra)  # [batch, N, extra_embedding_dim]
        
        # 4. Concatenate all features
        fused = torch.cat([t_features, c_features, extra_features], dim=-1)  # [batch, N, fused_dim]
        
        # 5. Optional projection
        output = self.proj(fused)  # [batch, N, output_dim]
        
        return output

