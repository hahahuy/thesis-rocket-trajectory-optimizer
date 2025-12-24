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
    
    # NOTE:
    # PINNLossV2 now reuses the reduced-order vertical physics loss implemented
    # in the base class (PINNLoss.physics_loss). The central-difference
    # derivative helper `compute_derivative` is kept for potential future
    # extensions but is not used in the current reduced-order formulation.

