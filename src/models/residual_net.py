"""
Hybrid residual network for rocket trajectory prediction.

This model uses a classical integrator baseline and predicts residual corrections.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable

from .architectures import MLP, TimeEmbedding, ContextEncoder


class ResidualNet(nn.Module):
    """
    Hybrid residual network that predicts corrections to a baseline trajectory.
    
    The baseline is typically from a classical integrator, and the network
    learns to predict residual corrections: state = baseline + residual.
    """
    
    def __init__(
        self,
        context_dim: int,
        n_hidden: int = 6,
        n_neurons: int = 128,
        activation: str = "tanh",
        fourier_features: int = 8,
        layer_norm: bool = True,
        dropout: float = 0.05,
        context_embedding_dim: int = 16,
        baseline_fn: Optional[Callable] = None
    ):
        super().__init__()
        
        self.context_dim = context_dim
        self.fourier_features = fourier_features
        self.baseline_fn = baseline_fn
        
        # Time embedding
        self.time_embedding = TimeEmbedding(n_frequencies=fourier_features)
        time_dim = 1 + 2 * fourier_features
        
        # Context encoder
        self.context_encoder = ContextEncoder(
            context_dim=context_dim,
            embedding_dim=context_embedding_dim,
            activation=activation
        )
        
        # Input dimension: time features + context embedding + baseline state
        input_dim = time_dim + context_embedding_dim + 14  # +14 for baseline state
        
        # MLP network (predicts residual)
        hidden_dims = [n_neurons] * n_hidden
        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=14,  # Residual state dimension
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout
        )
    
    def forward(
        self,
        t: torch.Tensor,
        context: torch.Tensor,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            t: Time [..., 1] or [batch, N, 1] (nondimensional)
            context: Context vector [..., context_dim] or [batch, context_dim]
            baseline: Baseline state [..., 14] (optional, computed if not provided)
            
        Returns:
            state: Predicted state [..., 14] = baseline + residual (nondimensional)
        """
        # Ensure t has correct shape
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        elif t.dim() == 2 and t.shape[-1] != 1:
            t = t.unsqueeze(-1)
        
        # Compute baseline if not provided
        if baseline is None:
            if self.baseline_fn is not None:
                baseline = self.baseline_fn(t, context)
            else:
                # Default: zero baseline
                if t.dim() == 2:
                    baseline = torch.zeros(t.shape[0], 14, device=t.device, dtype=t.dtype)
                else:
                    baseline = torch.zeros(*t.shape[:-1], 14, device=t.device, dtype=t.dtype)
        
        # Time embedding
        t_emb = self.time_embedding(t)  # [..., time_dim]
        
        # Context embedding
        if context.dim() == 2 and t.dim() == 3:
            batch_size, N = t.shape[:2]
            context = context.unsqueeze(1).expand(batch_size, N, -1)
        
        ctx_emb = self.context_encoder(context)  # [..., context_embedding_dim]
        
        # Concatenate: time + context + baseline
        x = torch.cat([t_emb, ctx_emb, baseline], dim=-1)
        
        # Predict residual
        residual = self.mlp(x)  # [..., 14]
        
        # Final state = baseline + residual
        state = baseline + residual
        
        return state
    
    def predict_trajectory(
        self,
        t: torch.Tensor,
        context: torch.Tensor,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict full trajectory for given time grid and context.
        
        Args:
            t: Time grid [N] (nondimensional)
            context: Context vector [context_dim]
            baseline: Baseline trajectory [N, 14] (optional)
            
        Returns:
            state: Trajectory [N, 14] (nondimensional)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [N, 1]
        
        if context.dim() == 1:
            context = context.unsqueeze(0)  # [1, context_dim]
        
        return self.forward(t, context, baseline).squeeze(0)  # [N, 14]

