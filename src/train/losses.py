"""
Loss functions for PINN training.

Includes data loss, physics loss (ODE residuals), and boundary condition loss.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from src.physics.dynamics_pytorch import compute_dynamics


class PINNLoss(nn.Module):
    """
    Combined loss for PINN training.
    
    L = λ_data * L_data + λ_phys * L_phys + λ_bc * L_bc
    """
    
    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_phys: float = 0.1,
        lambda_bc: float = 1.0,
        physics_params: Optional[Dict] = None,
        scales: Optional[Dict] = None
    ):
        super().__init__()
        
        self.lambda_data = lambda_data
        self.lambda_phys = lambda_phys
        self.lambda_bc = lambda_bc
        
        self.physics_params = physics_params or {}
        self.scales = scales or {}
        
        # Convert params to tensors (will be moved to device in forward)
        self._params_tensors = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in self.physics_params.items()
        }
    
    def data_loss(
        self,
        pred_state: torch.Tensor,
        true_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean squared error between predicted and true states.
        
        Args:
            pred_state: [batch, N, 14] or [N, 14]
            true_state: [batch, N, 14] or [N, 14]
            
        Returns:
            Scalar loss
        """
        return torch.mean((pred_state - true_state) ** 2)
    
    def physics_loss(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        control: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Physics loss: ODE residual norm.
        
        Computes ||s_dot - f(s, u, p)||^2 where f is the dynamics function.
        
        Args:
            t: Time [batch, N, 1] or [N, 1]
            state: State [batch, N, 14] or [N, 14]
            control: Control [batch, N, 4] or [N, 4] (optional, uses zero if None)
            
        Returns:
            Scalar loss
        """
        # Ensure correct shapes
        if state.dim() == 2:
            state = state.unsqueeze(0)  # [1, N, 14]
            t = t.unsqueeze(0)  # [1, N, 1]
            was_unbatched = True
        else:
            was_unbatched = False
        
        batch_size, N, state_dim = state.shape
        
        # Compute numerical derivative: ds/dt
        # Use central difference for interior points, forward/backward for boundaries
        # Work with [batch, N, state_dim] shape for correct indexing
        state_dot_numerical = torch.zeros_like(state)  # [batch, N, state_dim]
        
        # Extract time values per batch (assuming uniform time grid per batch)
        # t is [batch, N, 1], extract first batch's time for dt calculation
        t_batch0 = t[0, :, 0]  # [N] - time grid for first batch (should be same for all)
        
        for i in range(N):
            if i == 0:
                # Forward difference at first time step
                if N > 1:
                    dt = (t_batch0[i+1] - t_batch0[i]).item()
                    state_dot_numerical[:, i, :] = (
                        state[:, i+1, :] - state[:, i, :]
                    ) / (dt + 1e-12)
                else:
                    state_dot_numerical[:, i, :] = 0.0
            elif i == N - 1:
                # Backward difference at last time step
                dt = (t_batch0[i] - t_batch0[i-1]).item()
                state_dot_numerical[:, i, :] = (
                    state[:, i, :] - state[:, i-1, :]
                ) / (dt + 1e-12)
            else:
                # Central difference for interior points
                dt = (t_batch0[i+1] - t_batch0[i-1]).item()
                state_dot_numerical[:, i, :] = (
                    state[:, i+1, :] - state[:, i-1, :]
                ) / (dt + 1e-12)
        
        # Flatten for dynamics computation
        state_flat = state.view(-1, state_dim)  # [batch*N, 14]
        state_dot_numerical_flat = state_dot_numerical.view(-1, state_dim)  # [batch*N, 14]
        
        # Prepare control (use zero if not provided)
        # NOTE: Processed datasets (WP3) do not include control trajectories.
        # This is intentional: the PINN learns to predict states without explicit control,
        # as control is implicitly encoded in the trajectory. In WP5, control will become
        # an explicit model input for optimization. See WP4 docs for details.
        if control is None:
            # Default control: [T=0, theta_g=0, phi_g=0, delta=0] (zero thrust, no gimbal)
            control_flat = torch.zeros(batch_size * N, 4, device=state.device, dtype=state.dtype)
        else:
            if control.dim() == 2:
                control = control.unsqueeze(0)
            control_flat = control.view(-1, 4)
        
        # Move params to correct device
        params_device = {
            k: v.to(state.device) for k, v in self._params_tensors.items()
        }
        
        # Compute dynamics: f(s, u, p)
        state_dot_dynamics = compute_dynamics(
            state_flat,
            control_flat,
            params_device,
            self.scales
        )  # [batch*N, 14]
        
        # Residual: r = ds/dt_numerical - f(s, u, p)
        residual = state_dot_numerical_flat - state_dot_dynamics
        
        # Loss: mean squared residual
        loss = torch.mean(residual ** 2)
        
        return loss
    
    def boundary_loss(
        self,
        pred_state: torch.Tensor,
        true_state: torch.Tensor,
        t: torch.Tensor,
        t0: float = 0.0
    ) -> torch.Tensor:
        """
        Boundary condition loss: penalize mismatch at t=0 (and optionally t=T).
        
        Enforces boundary conditions on all batch elements, not just one sample.
        
        Args:
            pred_state: [batch, N, 14] or [N, 14]
            true_state: [batch, N, 14] or [N, 14]
            t: Time [batch, N, 1] or [N, 1]
            t0: Initial time (default 0.0)
            
        Returns:
            Scalar loss
        """
        # Ensure batched format
        if pred_state.dim() == 2:
            pred_state = pred_state.unsqueeze(0)  # [1, N, 14]
            true_state = true_state.unsqueeze(0)  # [1, N, 14]
            t = t.unsqueeze(0)  # [1, N, 1]
        
        batch_size, N, state_dim = pred_state.shape
        
        # Find time index closest to t0 (time grid should be same for all batches)
        # Use first batch's time grid to find index
        t_batch0 = t[0, :, 0]  # [N]
        time_idx = torch.argmin(torch.abs(t_batch0 - t0)).item()
        
        # Extract initial states for all batches
        pred_init = pred_state[:, time_idx, :]  # [batch, 14]
        true_init = true_state[:, time_idx, :]  # [batch, 14]
        
        # Compute MSE across all batch elements
        return torch.mean((pred_init - true_init) ** 2)
    
    def forward(
        self,
        pred_state: torch.Tensor,
        true_state: torch.Tensor,
        t: torch.Tensor,
        control: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss and component losses.
        
        Args:
            pred_state: [batch, N, 14] or [N, 14]
            true_state: [batch, N, 14] or [N, 14]
            t: Time [batch, N, 1] or [N, 1]
            control: Control [batch, N, 4] or [N, 4] (optional)
            
        Returns:
            (total_loss, loss_dict) where loss_dict contains component losses
        """
        L_data = self.data_loss(pred_state, true_state)
        L_phys = self.physics_loss(t, pred_state, control)
        L_bc = self.boundary_loss(pred_state, true_state, t)
        
        total_loss = (
            self.lambda_data * L_data +
            self.lambda_phys * L_phys +
            self.lambda_bc * L_bc
        )
        
        loss_dict = {
            "total": total_loss,
            "data": L_data,
            "physics": L_phys,
            "boundary": L_bc
        }
        
        return total_loss, loss_dict

