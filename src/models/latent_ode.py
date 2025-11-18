"""
[PINN_V2][2025-01-XX][Direction A]
Physics-Informed Neural Network with Latent Neural ODE.

This module implements Direction A from PINN_v2_Refactor_Directions.md:
- LatentDynamicsNet: Neural network that computes dz/dt in latent space
- LatentODEBlock: Fixed-step Euler integrator for latent ODE
- RocketLatentODEPINN: Main model that encodes context to z0, evolves through ODE, decodes to state

Change: Architecture redesign to improve RMSE and stability by aligning with ODE structure.
"""

import torch
import torch.nn as nn
from typing import Optional

from .architectures import MLP, TimeEmbedding, ContextEncoder


class LatentDynamicsNet(nn.Module):
    """
    [PINN_V2][2025-01-XX][Direction A]
    Neural network that computes latent state derivative dz/dt.
    
    Input: latent z + time embedding + context embedding
    Output: dz/dt
    """
    
    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        n_hidden: int = 3,
        n_neurons: int = 128,
        activation: str = "tanh",
        layer_norm: bool = True,
        dropout: float = 0.05
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # Input: [z || t_emb || ctx_emb]
        input_dim = latent_dim + condition_dim
        
        hidden_dims = [n_neurons] * n_hidden
        self.network = MLP(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout
        )
    
    def forward(
        self,
        z: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dz/dt = g_θ(z, condition).
        
        Args:
            z: Latent state [..., latent_dim]
            condition: Concatenated [t_emb || ctx_emb] [..., condition_dim]
            
        Returns:
            dz_dt: Latent derivative [..., latent_dim]
        """
        x = torch.cat([z, condition], dim=-1)
        dz_dt = self.network(x)
        return dz_dt


class LatentODEBlock(nn.Module):
    """
    [PINN_V2][2025-01-XX][Direction A]
    Fixed-step Euler integrator for latent ODE.
    
    Integrates: dz/dt = g_θ(z, t, ctx_emb) using Euler method.
    """
    
    def __init__(self, dynamics_net: LatentDynamicsNet):
        super().__init__()
        self.dynamics_net = dynamics_net
    
    def forward(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate latent ODE from z0 over time grid t.
        
        Args:
            z0: Initial latent state [batch, latent_dim]
            t: Time grid [batch, N, 1] or [N, 1]
            condition: Condition vectors [batch, N, condition_dim] or [N, condition_dim]
            
        Returns:
            z_traj: Latent trajectory [batch, N, latent_dim] or [N, latent_dim]
        """
        # Ensure batched format
        if t.dim() == 2:
            t = t.unsqueeze(0)  # [1, N, 1]
            condition = condition.unsqueeze(0)  # [1, N, condition_dim]
            z0 = z0.unsqueeze(0)  # [1, latent_dim]
            was_unbatched = True
        else:
            was_unbatched = False
        
        batch_size, N, _ = t.shape
        latent_dim = z0.shape[-1]
        
        # Initialize trajectory
        z_traj = torch.zeros(batch_size, N, latent_dim, device=z0.device, dtype=z0.dtype)
        z_traj[:, 0, :] = z0
        
        # Euler integration
        z_current = z0  # [batch, latent_dim]
        
        for i in range(N - 1):
            # Current condition: [batch, condition_dim]
            cond_i = condition[:, i, :]
            
            # Compute derivative
            dz_dt = self.dynamics_net(z_current, cond_i)  # [batch, latent_dim]
            
            # Time step
            dt = (t[:, i+1, 0] - t[:, i, 0]).unsqueeze(-1)  # [batch, 1]
            
            # Euler step: z_{i+1} = z_i + dt * dz_dt
            z_next = z_current + dt * dz_dt
            z_traj[:, i+1, :] = z_next
            z_current = z_next
        
        if was_unbatched:
            z_traj = z_traj.squeeze(0)  # [N, latent_dim]
        
        return z_traj


class RocketLatentODEPINN(nn.Module):
    """
    [PINN_V2][2025-01-XX][Direction A]
    Physics-Informed Neural Network with Latent Neural ODE for rocket trajectories.
    
    Architecture:
    1. Encode context → z0 (latent initial state)
    2. Evolve z(t) through neural ODE: dz/dt = g_θ(z, t, ctx_emb)
    3. Decode z(t) → s(t) (physical state)
    
    Input: (t, context) -> Output: state [14]
    """
    
    def __init__(
        self,
        context_dim: int,
        latent_dim: int = 64,
        context_embedding_dim: int = 64,
        fourier_features: int = 8,
        dynamics_n_hidden: int = 3,
        dynamics_n_neurons: int = 128,
        decoder_n_hidden: int = 3,
        decoder_n_neurons: int = 128,
        activation: str = "tanh",
        layer_norm: bool = True,
        dropout: float = 0.05
    ):
        super().__init__()
        
        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.fourier_features = fourier_features
        
        # Time embedding: t -> [t, sin(2πk t), cos(2πk t)]
        self.time_embedding = TimeEmbedding(n_frequencies=fourier_features)
        time_dim = 1 + 2 * fourier_features
        
        # Context encoder for embedding
        self.context_encoder = ContextEncoder(
            context_dim=context_dim,
            embedding_dim=context_embedding_dim,
            activation=activation
        )
        
        # Encoder: context embedding → z0 (latent initial state)
        self.z0_encoder = nn.Sequential(
            nn.Linear(context_embedding_dim, latent_dim),
            nn.Tanh()
        )
        
        # Condition dimension: time_emb + context_emb
        condition_dim = time_dim + context_embedding_dim
        
        # Latent dynamics network: dz/dt = g_θ(z, condition)
        self.dynamics_net = LatentDynamicsNet(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            n_hidden=dynamics_n_hidden,
            n_neurons=dynamics_n_neurons,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout
        )
        
        # ODE integrator
        self.ode_block = LatentODEBlock(self.dynamics_net)
        
        # Decoder: latent z → physical state s
        hidden_dims = [decoder_n_neurons] * decoder_n_hidden
        self.decoder = MLP(
            input_dim=latent_dim,
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
            t = t.unsqueeze(-1)  # [N, 1]
        elif t.dim() == 2 and t.shape[-1] != 1:
            t = t.unsqueeze(-1)
        
        # Track if we need to add batch dimension
        t_was_unbatched = (t.dim() == 2)  # [N, 1] without batch
        context_was_unbatched = (context.dim() == 1)  # [context_dim] without batch
        
        # Ensure context is batched
        if context.dim() == 1:
            context = context.unsqueeze(0)  # [1, context_dim]
        
        # Ensure t is batched if context is batched
        if t.dim() == 2 and context.dim() == 2:
            # t: [N, 1], context: [1, context_dim] -> add batch to t
            t = t.unsqueeze(0)  # [1, N, 1]
        
        # Context embedding
        if context.dim() == 2 and t.dim() == 3:
            # context: [batch, context_dim], t: [batch, N, 1]
            batch_size, N = t.shape[:2]
            # Encode context once per batch
            ctx_emb_batch = self.context_encoder(context)  # [batch, context_embedding_dim]
            # Broadcast over time steps
            ctx_emb = ctx_emb_batch.unsqueeze(1).expand(batch_size, N, -1)  # [batch, N, context_embedding_dim]
        else:
            ctx_emb = self.context_encoder(context)  # [..., context_embedding_dim]
        
        # Encode context to initial latent state z0
        if ctx_emb.dim() == 3:
            # Use first time step's context embedding (or average)
            ctx_emb_for_z0 = ctx_emb[:, 0, :]  # [batch, context_embedding_dim]
        else:
            ctx_emb_for_z0 = ctx_emb
        
        z0 = self.z0_encoder(ctx_emb_for_z0)  # [batch, latent_dim] or [latent_dim]
        
        # Time embedding
        t_emb = self.time_embedding(t)  # [..., time_dim]
        
        # Prepare condition: concatenate time and context embeddings
        if t_emb.dim() == 3 and ctx_emb.dim() == 3:
            condition = torch.cat([t_emb, ctx_emb], dim=-1)  # [batch, N, condition_dim]
        else:
            condition = torch.cat([t_emb, ctx_emb], dim=-1)  # [..., condition_dim]
        
        # Integrate latent ODE
        z_traj = self.ode_block(z0, t, condition)  # [batch, N, latent_dim] or [N, latent_dim]
        
        # Decode to physical state
        state = self.decoder(z_traj)  # [batch, N, 14] or [N, 14]
        
        # Remove batch dimension if inputs were unbatched
        if t_was_unbatched and context_was_unbatched and state.dim() == 3:
            state = state.squeeze(0)  # [N, 14]
        
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

