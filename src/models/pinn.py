"""
Physics-Informed Neural Network (PINN) for rocket trajectory prediction.

The model takes time and context parameters as input and predicts the full
6-DOF state vector. Physics loss is computed via autograd on the dynamics.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .architectures import MLP, TimeEmbedding, ContextEncoder


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for 6-DOF rocket trajectories.
    
    Input: (t, context) -> Output: state [14]
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
        context_embedding_dim: int = 16
    ):
        super().__init__()
        
        self.context_dim = context_dim
        self.fourier_features = fourier_features
        
        # Time embedding: t -> [t, sin(2πk t), cos(2πk t)]
        self.time_embedding = TimeEmbedding(n_frequencies=fourier_features)
        time_dim = 1 + 2 * fourier_features
        
        # Context encoder
        self.context_encoder = ContextEncoder(
            context_dim=context_dim,
            embedding_dim=context_embedding_dim,
            activation=activation
        )
        
        # Input dimension: time features + context embedding
        input_dim = time_dim + context_embedding_dim
        
        # MLP network
        hidden_dims = [n_neurons] * n_hidden
        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=14,  # State dimension
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout
        )
    
    def forward(
        self,
        t: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            t: Time [..., 1] or [batch, N, 1] (nondimensional)
            context: Context vector [..., context_dim] or [batch, context_dim]
            
        Returns:
            state: Predicted state [..., 14] or [batch, N, 14] (nondimensional)
        """
        # Ensure t has correct shape
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        elif t.dim() == 2 and t.shape[-1] != 1:
            t = t.unsqueeze(-1)
        
        # Time embedding
        t_emb = self.time_embedding(t)  # [..., time_dim]
        
        # Context embedding
        # If context is [batch, context_dim] but t is [batch, N, 1], broadcast
        if context.dim() == 2 and t.dim() == 3:
            # context: [batch, context_dim], t: [batch, N, 1]
            batch_size, N = t.shape[:2]
            context = context.unsqueeze(1).expand(batch_size, N, -1)
        
        ctx_emb = self.context_encoder(context)  # [..., context_embedding_dim]
        
        # Concatenate
        x = torch.cat([t_emb, ctx_emb], dim=-1)  # [..., time_dim + context_embedding_dim]
        
        # MLP
        state = self.mlp(x)  # [..., 14]
        
        return state
    
    def predict_trajectory(
        self,
        t: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict full trajectory for given time grid and context.
        
        Args:
            t: Time grid [N] (nondimensional)
            context: Context vector [context_dim]
            
        Returns:
            state: Trajectory [N, 14] (nondimensional)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [N, 1]
        
        if context.dim() == 1:
            context = context.unsqueeze(0)  # [1, context_dim]
        
        return self.forward(t, context).squeeze(0)  # [N, 14]

