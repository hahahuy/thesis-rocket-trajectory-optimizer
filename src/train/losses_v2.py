# =========================
# PINN Loss V2 (Central Diff)
# =========================

import torch
from typing import Dict, Optional, Tuple

from src.train.losses import PINNLoss
from src.physics.derivatives_v2 import central_difference
from src.physics.dynamics_pytorch import compute_dynamics


class PINNLossV2(PINNLoss):
    """
    PINN Loss with central difference derivative computation (v2).
    
    This class extends PINNLoss and overrides the derivative computation
    method to use central difference instead of forward difference.
    
    The central difference method:
        ds/dt = (s(t+1) - s(t-1)) / (2*dt)
    
    provides smoother and more accurate derivatives, especially for
    accelerations and rotations, leading to improved physics residual accuracy.
    
    Enhanced with:
    - Component scaling for physics residuals (pos, vel, quat, ang, mass)
    - Reweighted physics terms for better balance across state components
    """
    
    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_phys: float = 0.1,
        lambda_bc: float = 1.0,
        physics_params: Optional[Dict] = None,
        scales: Optional[Dict] = None,
        # Enhanced loss parameters (inherited from PINNLoss)
        component_weights: Optional[Dict[str, float]] = None,
        lambda_quat_norm: float = 0.0,
        lambda_mass_flow: float = 0.0,
        lambda_translation: float = 1.0,
        lambda_rotation: float = 1.0,
        lambda_mass: float = 1.0,
        # Soft physics + smoothing (Direction D1.5)
        lambda_mass_residual: float = 0.0,
        lambda_vz_residual: float = 0.0,
        lambda_vxy_residual: float = 0.0,
        lambda_smooth_z: float = 0.0,
        lambda_smooth_vz: float = 0.0,
        # Position-Velocity consistency + position smoothing (Direction D1.51)
        lambda_pos_vel: float = 0.0,
        lambda_smooth_pos: float = 0.0,
        # Horizontal motion suppression (Direction D1.52)
        lambda_zero_vxy: float = 0.0,
        lambda_zero_axy: float = 0.0,
        lambda_hacc: float = 0.0,
        lambda_xy_zero: float = 0.0,
        # V2 specific: Component scaling and reweighted physics
        physics_scale: Optional[Dict[str, float]] = None,
        physics_groups: Optional[Dict[str, float]] = None,
    ):
        super().__init__(
            lambda_data=lambda_data,
            lambda_phys=lambda_phys,
            lambda_bc=lambda_bc,
            physics_params=physics_params,
            scales=scales,
            component_weights=component_weights,
            lambda_quat_norm=lambda_quat_norm,
            lambda_mass_flow=lambda_mass_flow,
            lambda_translation=lambda_translation,
            lambda_rotation=lambda_rotation,
            lambda_mass=lambda_mass,
            lambda_mass_residual=lambda_mass_residual,
            lambda_vz_residual=lambda_vz_residual,
            lambda_vxy_residual=lambda_vxy_residual,
            lambda_smooth_z=lambda_smooth_z,
            lambda_smooth_vz=lambda_smooth_vz,
            lambda_pos_vel=lambda_pos_vel,
            lambda_smooth_pos=lambda_smooth_pos,
            lambda_zero_vxy=lambda_zero_vxy,
            lambda_zero_axy=lambda_zero_axy,
            lambda_hacc=lambda_hacc,
            lambda_xy_zero=lambda_xy_zero,
        )
        
        # Default component scales for physics residuals
        default_physics_scale = {
            'pos': 1.0,
            'vel': 1.0,
            'quat': 0.1,
            'ang': 0.2,
            'mass': 1e-3
        }
        if physics_scale:
            default_physics_scale.update(physics_scale)
        self.physics_scale = default_physics_scale
        
        # Default weights for physics groups
        default_physics_groups = {
            'pos': 1.0,
            'vel': 1.0,
            'quat': 0.2,
            'ang': 0.5,
            'mass': 1.0
        }
        if physics_groups:
            default_physics_groups.update(physics_groups)
        self.physics_groups = default_physics_groups
    
    def compute_derivative(
        self,
        state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Override the v1 forward-difference with v2 central difference.
        
        Args:
            state: [batch, N, dim] - state trajectory
            t: [batch, N, 1] - time grid
            
        Returns:
            state_dot: [batch, N, dim] - time derivative of state
        """
        # Use central difference (handles non-uniform time grids)
        return central_difference(state, t, eps=self._eps)
    
    def physics_loss(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        control: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Physics loss: ODE residual norm (v2 with central difference).
        
        Computes ||s_dot - f(s, u, p)||^2 where f is the dynamics function.
        Uses central difference for derivative computation.
        
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
        
        # Compute numerical derivative using central difference (v2)
        state_dot_numerical = self.compute_derivative(state, t)  # [batch, N, state_dim]
        
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
        residual = state_dot_numerical_flat - state_dot_dynamics  # [batch*N, 14]
        
        # Reshape back to [batch, N, 14] for component-wise processing
        residual = residual.view(batch_size, N, state_dim)  # [batch, N, 14]
        
        # Extract component residuals
        # State order: [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]
        r_pos = residual[..., 0:3]  # [batch, N, 3] - position (x, y, z)
        r_vel = residual[..., 3:6]  # [batch, N, 3] - velocity (vx, vy, vz)
        r_quat = residual[..., 6:10]  # [batch, N, 4] - quaternion (q0, q1, q2, q3)
        r_ang = residual[..., 10:13]  # [batch, N, 3] - angular velocity (wx, wy, wz)
        r_m = residual[..., 13:14]  # [batch, N, 1] - mass
        
        # Apply component scaling and compute weighted losses
        # L_phys = sum(w[group] * (r_group / scale[group])^2)
        scale_pos = self.physics_scale['pos']
        scale_vel = self.physics_scale['vel']
        scale_quat = self.physics_scale['quat']
        scale_ang = self.physics_scale['ang']
        scale_mass = self.physics_scale['mass']
        
        w_pos = self.physics_groups['pos']
        w_vel = self.physics_groups['vel']
        w_quat = self.physics_groups['quat']
        w_ang = self.physics_groups['ang']
        w_mass = self.physics_groups['mass']
        
        # Compute scaled and weighted losses
        L_pos = w_pos * (r_pos / scale_pos).pow(2).mean()
        L_vel = w_vel * (r_vel / scale_vel).pow(2).mean()
        L_quat = w_quat * (r_quat / scale_quat).pow(2).mean()
        L_ang = w_ang * (r_ang / scale_ang).pow(2).mean()
        L_mass = w_mass * (r_m / scale_mass).pow(2).mean()
        
        # Total physics loss
        loss = L_pos + L_vel + L_quat + L_ang + L_mass
        
        return loss

