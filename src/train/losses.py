"""
Loss functions for PINN training.

Includes data loss, physics loss (ODE residuals), and boundary condition loss.
Enhanced with component-weighted loss, quaternion normalization penalty,
and mass flow consistency constraints.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from src.physics.dynamics_pytorch import compute_dynamics


class PINNLoss(nn.Module):
    """
    Combined loss for PINN training.
    
    L = λ_data * L_data + λ_phys * L_phys + λ_bc * L_bc + 
        λ_quat_norm * L_quat_norm + λ_mass_flow * L_mass_flow
    
    Enhanced with:
    - Component-weighted data loss (prioritize z, vz, quaternions)
    - Quaternion normalization penalty
    - Mass flow consistency (non-increasing mass)
    - Group weights for translation/rotation/mass
    """
    
    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_phys: float = 0.1,
        lambda_bc: float = 1.0,
        physics_params: Optional[Dict] = None,
        scales: Optional[Dict] = None,
        # Enhanced loss parameters
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
    ):
        super().__init__()
        
        self.lambda_data = lambda_data
        self.lambda_phys = lambda_phys
        self.lambda_bc = lambda_bc
        self.lambda_quat_norm = lambda_quat_norm
        self.lambda_mass_flow = lambda_mass_flow
        self.lambda_translation = lambda_translation
        self.lambda_rotation = lambda_rotation
        self.lambda_mass = lambda_mass
        self.lambda_mass_residual = lambda_mass_residual
        self.lambda_vz_residual = lambda_vz_residual
        self.lambda_vxy_residual = lambda_vxy_residual
        self.lambda_smooth_z = lambda_smooth_z
        self.lambda_smooth_vz = lambda_smooth_vz
        self.lambda_pos_vel = lambda_pos_vel
        self.lambda_smooth_pos = lambda_smooth_pos
        self.lambda_zero_vxy = lambda_zero_vxy
        self.lambda_zero_axy = lambda_zero_axy
        self.lambda_hacc = lambda_hacc
        self.lambda_xy_zero = lambda_xy_zero
        
        self.physics_params = physics_params or {}
        self.scales = scales or {}
        
        # Component weights for state variables
        # Default: uniform weights, but can be customized
        # State order: [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]
        default_weights = {
            "x": 1.0, "y": 1.0, "z": 1.0,
            "vx": 1.0, "vy": 1.0, "vz": 1.0,
            "q0": 1.0, "q1": 1.0, "q2": 1.0, "q3": 1.0,
            "wx": 1.0, "wy": 1.0, "wz": 1.0,
            "m": 1.0,
        }
        if component_weights:
            default_weights.update(component_weights)
        
        # Convert to tensor for efficient computation
        # State indices: [x=0, y=1, z=2, vx=3, vy=4, vz=5, q0=6, q1=7, q2=8, q3=9, wx=10, wy=11, wz=12, m=13]
        state_order = ["x", "y", "z", "vx", "vy", "vz", "q0", "q1", "q2", "q3", "wx", "wy", "wz", "m"]
        self.register_buffer(
            "component_weights_tensor",
            torch.tensor([default_weights[k] for k in state_order], dtype=torch.float32)
        )
        
        # Convert params to tensors (will be moved to device in forward)
        self._params_tensors = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in self.physics_params.items()
        }

        self._eps = 1e-8
    
    def data_loss(
        self,
        pred_state: torch.Tensor,
        true_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Component-weighted mean squared error between predicted and true states.
        
        Args:
            pred_state: [batch, N, 14] or [N, 14]
            true_state: [batch, N, 14] or [N, 14]
            
        Returns:
            Scalar loss
        """
        # Ensure batched format
        if pred_state.dim() == 2:
            pred_state = pred_state.unsqueeze(0)
            true_state = true_state.unsqueeze(0)
        
        # Compute squared errors per component: [batch, N, 14]
        squared_errors = (pred_state - true_state) ** 2
        
        # Apply component weights: [14] -> [1, 1, 14]
        weights = self.component_weights_tensor.view(1, 1, -1).to(pred_state.device)
        weighted_errors = squared_errors * weights
        
        # Separate into translation, rotation, mass groups
        # Translation: [x, y, z, vx, vy, vz] = indices [0:6]
        # Rotation: [q0, q1, q2, q3, wx, wy, wz] = indices [6:13]
        # Mass: [m] = index [13]
        translation_loss = torch.mean(weighted_errors[:, :, 0:6]) * self.lambda_translation
        rotation_loss = torch.mean(weighted_errors[:, :, 6:13]) * self.lambda_rotation
        mass_loss = torch.mean(weighted_errors[:, :, 13:14]) * self.lambda_mass
        
        return translation_loss + rotation_loss + mass_loss

    def _broadcast_context(
        self,
        context: Optional[torch.Tensor],
        batch_size: int,
        N: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if context is None:
            return None
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(batch_size, N, -1)
        elif context.dim() == 3 and context.shape[1] == 1:
            context = context.expand(batch_size, N, -1)
        return context.to(device)

    def _finite_difference(
        self,
        values: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # values: [batch, N, d], t: [batch, N, 1]
        batch, N, d = values.shape
        if N < 2:
            return torch.zeros_like(values)
        diffs = torch.zeros_like(values)
        dt = t[:, 1:, 0] - t[:, :-1, 0]  # [batch, N-1]
        dt = dt.clamp_min(self._eps).unsqueeze(-1)
        diffs[:, :-1, :] = (values[:, 1:, :] - values[:, :-1, :]) / dt
        diffs[:, -1, :] = diffs[:, -2, :]
        return diffs

    def _second_difference(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        if values.shape[1] < 3:
            return torch.zeros(1, device=values.device, dtype=values.dtype)
        second = values[:, 2:, :] - 2.0 * values[:, 1:-1, :] + values[:, :-2, :]
        return torch.mean(second ** 2)

    def _position_velocity_consistency_loss(
        self,
        state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Position-Velocity Consistency Loss (PV-Loss).
        
        Enforces discrete motion update: p(t+1) ≈ p(t) + v(t) * Δt
        
        Args:
            state: [batch, N, 14] - predicted state
            t: [batch, N, 1] - time grid
            
        Returns:
            Scalar loss
        """
        # Extract positions: [batch, N, 3] (x, y, z)
        positions = state[..., 0:3]  # [batch, N, 3]
        # Extract velocities: [batch, N, 3] (vx, vy, vz)
        velocities = state[..., 3:6]  # [batch, N, 3]
        
        if positions.shape[1] < 2:
            return torch.tensor(0.0, device=state.device, dtype=state.dtype)
        
        # Compute time differences: [batch, N-1]
        dt = t[:, 1:, 0] - t[:, :-1, 0]  # [batch, N-1]
        dt = dt.clamp_min(self._eps).unsqueeze(-1)  # [batch, N-1, 1]
        
        # Compute residuals: r = p(t+1) - p(t) - v(t) * Δt
        # positions[:, 1:, :] - positions[:, :-1, :] gives p(t+1) - p(t)
        # velocities[:, :-1, :] * dt gives v(t) * Δt
        residual = (
            positions[:, 1:, :] - positions[:, :-1, :] - velocities[:, :-1, :] * dt
        )  # [batch, N-1, 3]
        
        # Loss: mean of squared residuals across all components
        loss = torch.mean(residual ** 2)
        return loss

    def _position_smoothing_loss(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Position Smoothing Loss (3D).
        
        Penalizes high-frequency curvature in X/Y/Z trajectories using second-order
        discrete derivative (curvature): c(t) = x(t+1) - 2*x(t) + x(t-1)
        
        Args:
            state: [batch, N, 14] - predicted state
            
        Returns:
            Scalar loss (for x, y, z - all 3D position components)
        """
        # Extract all position components: [batch, N, 3] (x, y, z)
        positions = state[..., 0:3]  # [batch, N, 3]
        
        if positions.shape[1] < 3:
            return torch.tensor(0.0, device=state.device, dtype=state.dtype)
        
        # Compute curvature: c(t) = x(t+1) - 2*x(t) + x(t-1)
        # For indices 1..N-2: curvature[:, i] = pos[:, i+1] - 2*pos[:, i] + pos[:, i-1]
        curvature = (
            positions[:, 2:, :] - 2.0 * positions[:, 1:-1, :] + positions[:, :-2, :]
        )  # [batch, N-2, 3]
        
        # Loss: mean of squared curvature
        loss = torch.mean(curvature ** 2)
        return loss

    def _zero_horizontal_velocity_loss(
        self,
        pred_state: torch.Tensor,
        true_state: torch.Tensor,
        v_thresh: float = 1e-3,
    ) -> torch.Tensor:
        """
        Zero horizontal velocity loss (D1.52).
        
        Forces predicted horizontal velocities to be near zero where true
        horizontal velocities are near zero (vertical-only segments).
        
        Args:
            pred_state: [batch, N, 14] - predicted state
            true_state: [batch, N, 14] - true state
            v_thresh: Threshold for detecting zero horizontal velocity
            
        Returns:
            Scalar loss
        """
        # Ensure batched format
        if pred_state.dim() == 2:
            pred_state = pred_state.unsqueeze(0)
            true_state = true_state.unsqueeze(0)
        
        # Extract horizontal velocities
        vx_pred = pred_state[..., 3:4]  # [batch, N, 1]
        vy_pred = pred_state[..., 4:5]  # [batch, N, 1]
        vx_true = true_state[..., 3:4]  # [batch, N, 1]
        vy_true = true_state[..., 4:5]  # [batch, N, 1]
        
        # Create mask: 1 where horizontal true velocity is (almost) zero
        mask_zero_xy = (
            (vx_true.abs() < v_thresh) & (vy_true.abs() < v_thresh)
        ).float()  # [batch, N, 1]
        
        # Loss: penalize predicted horizontal velocity where mask is active
        loss = (mask_zero_xy * (vx_pred**2 + vy_pred**2)).mean()
        return loss

    def _zero_horizontal_acceleration_loss(
        self,
        pred_state: torch.Tensor,
        true_state: torch.Tensor,
        t: torch.Tensor,
        v_thresh: float = 1e-3,
    ) -> torch.Tensor:
        """
        Zero horizontal acceleration loss (D1.52).
        
        Forces predicted horizontal accelerations to be near zero where true
        horizontal velocities are near zero (vertical-only segments).
        
        Args:
            pred_state: [batch, N, 14] - predicted state
            true_state: [batch, N, 14] - true state
            t: [batch, N, 1] - time grid
            v_thresh: Threshold for detecting zero horizontal velocity
            
        Returns:
            Scalar loss
        """
        # Ensure batched format
        if pred_state.dim() == 2:
            pred_state = pred_state.unsqueeze(0)
            true_state = true_state.unsqueeze(0)
            t = t.unsqueeze(0)
        
        if pred_state.shape[1] < 2:
            return torch.tensor(0.0, device=pred_state.device, dtype=pred_state.dtype)
        
        # Extract horizontal velocities
        vx_pred = pred_state[..., 3:4]  # [batch, N, 1]
        vy_pred = pred_state[..., 4:5]  # [batch, N, 1]
        vx_true = true_state[..., 3:4]  # [batch, N, 1]
        vy_true = true_state[..., 4:5]  # [batch, N, 1]
        
        # Compute time differences
        dt = t[:, 1:, 0] - t[:, :-1, 0]  # [batch, N-1]
        dt = dt.clamp_min(self._eps).unsqueeze(-1)  # [batch, N-1, 1]
        
        # Compute predicted horizontal acceleration from finite differences
        vx_mid = vx_pred[:, 1:, :] - vx_pred[:, :-1, :]  # [batch, N-1, 1]
        vy_mid = vy_pred[:, 1:, :] - vy_pred[:, :-1, :]  # [batch, N-1, 1]
        ax_pred = vx_mid / dt  # [batch, N-1, 1]
        ay_pred = vy_mid / dt  # [batch, N-1, 1]
        
        # Create mask for midpoints: both endpoints should have zero horizontal velocity
        mask_zero_xy = (
            (vx_true.abs() < v_thresh) & (vy_true.abs() < v_thresh)
        ).float()  # [batch, N, 1]
        mask_mid = mask_zero_xy[:, 1:, :] * mask_zero_xy[:, :-1, :]  # [batch, N-1, 1]
        
        # Loss: penalize predicted horizontal acceleration where mask is active
        loss = (mask_mid * (ax_pred**2 + ay_pred**2)).mean()
        return loss

    def _global_horizontal_acceleration_loss(
        self,
        pred_state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Global horizontal acceleration limiter (D1.52).
        
        Keeps lateral motion small and smooth across all cases, without
        forcing it to zero. Applied globally (no masking).
        
        Args:
            pred_state: [batch, N, 14] - predicted state
            t: [batch, N, 1] - time grid
            
        Returns:
            Scalar loss
        """
        # Ensure batched format
        if pred_state.dim() == 2:
            pred_state = pred_state.unsqueeze(0)
            t = t.unsqueeze(0)
        
        if pred_state.shape[1] < 2:
            return torch.tensor(0.0, device=pred_state.device, dtype=pred_state.dtype)
        
        # Extract horizontal velocities
        vx_pred = pred_state[..., 3:4]  # [batch, N, 1]
        vy_pred = pred_state[..., 4:5]  # [batch, N, 1]
        
        # Compute time differences
        dt = t[:, 1:, 0] - t[:, :-1, 0]  # [batch, N-1]
        dt = dt.clamp_min(self._eps).unsqueeze(-1)  # [batch, N-1, 1]
        
        # Compute predicted horizontal acceleration from finite differences
        vx_mid = vx_pred[:, 1:, :] - vx_pred[:, :-1, :]  # [batch, N-1, 1]
        vy_mid = vy_pred[:, 1:, :] - vy_pred[:, :-1, :]  # [batch, N-1, 1]
        ax_pred = vx_mid / dt  # [batch, N-1, 1]
        ay_pred = vy_mid / dt  # [batch, N-1, 1]
        
        # Loss: mean squared horizontal acceleration (no masking)
        loss = (ax_pred**2 + ay_pred**2).mean()
        return loss

    def _xy_zero_position_loss(
        self,
        pred_state: torch.Tensor,
        true_state: torch.Tensor,
        v_thresh: float = 1e-3,
    ) -> torch.Tensor:
        """
        Upweighted x/y position error when true x/y ≈ 0 (D1.52).
        
        Tells the model that for vertical-only timesteps, any lateral
        displacement is expensive.
        
        Args:
            pred_state: [batch, N, 14] - predicted state
            true_state: [batch, N, 14] - true state
            v_thresh: Threshold for detecting zero horizontal velocity
            
        Returns:
            Scalar loss
        """
        # Ensure batched format
        if pred_state.dim() == 2:
            pred_state = pred_state.unsqueeze(0)
            true_state = true_state.unsqueeze(0)
        
        # Extract positions
        x_pred = pred_state[..., 0:1]  # [batch, N, 1]
        y_pred = pred_state[..., 1:2]  # [batch, N, 1]
        x_true = true_state[..., 0:1]  # [batch, N, 1]
        y_true = true_state[..., 1:2]  # [batch, N, 1]
        
        # Extract horizontal velocities for masking
        vx_true = true_state[..., 3:4]  # [batch, N, 1]
        vy_true = true_state[..., 4:5]  # [batch, N, 1]
        
        # Create mask: 1 where horizontal true velocity is (almost) zero
        mask_zero_xy = (
            (vx_true.abs() < v_thresh) & (vy_true.abs() < v_thresh)
        ).float()  # [batch, N, 1]
        
        # Loss: penalize x/y position error where mask is active
        loss = (mask_zero_xy * ((x_pred - x_true)**2 + (y_pred - y_true)**2)).mean()
        return loss

    def _get_param(self, name: str, default: float) -> torch.Tensor:
        if name in self._params_tensors:
            return self._params_tensors[name]
        return torch.tensor(default, dtype=torch.float32)

    def _mass_residual_loss(
        self,
        state: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        # state: [batch, N, 14]
        m = state[..., 13:14]
        dm_dt = self._finite_difference(m, t)

        Isp = context[..., 1:2]
        Tmax = context[..., 5:6]
        g0 = self._get_param("g0", 9.81).to(state.device)
        g0 = g0.view(1, 1, 1)

        expected = Tmax / (Isp.clamp_min(self._eps) * g0)
        residual = dm_dt + expected
        return torch.mean(residual ** 2)

    def _drag_acc_component(
        self,
        velocity_component: torch.Tensor,
        rho: torch.Tensor,
        Cd: torch.Tensor,
        S_ref: torch.Tensor,
        mass: torch.Tensor,
    ) -> torch.Tensor:
        drag_force = 0.5 * rho * Cd * S_ref * velocity_component * torch.abs(velocity_component)
        drag_acc = drag_force / mass.clamp_min(self._eps)
        return drag_acc

    def _compute_density(self, altitude: torch.Tensor) -> torch.Tensor:
        rho0 = self._get_param("rho0", 1.225).to(altitude.device).view(1, 1, 1)
        h_scale = self._get_param("h_scale", 8400.0).to(altitude.device).view(1, 1, 1)
        return rho0 * torch.exp(-torch.clamp(altitude, min=0.0) / h_scale)

    def _vertical_residual_loss(
        self,
        state: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        v = state[..., 3:6]
        vz = v[..., 2:3]
        dvz_dt = self._finite_difference(vz, t)

        z = state[..., 2:3]
        rho = self._compute_density(z)
        Cd = context[..., 2:3]
        S_ref = self._get_param("S_ref", 0.05).to(state.device).view(1, 1, 1)
        mass = state[..., 13:14]

        drag_acc_z = self._drag_acc_component(vz, rho, Cd, S_ref, mass)

        g0 = self._get_param("g0", 9.81).to(state.device).view(1, 1, 1)
        Tmax = context[..., 5:6]
        thrust_acc = Tmax / mass.clamp_min(self._eps)
        a_phys = thrust_acc - g0 - drag_acc_z
        residual = dvz_dt - a_phys
        return torch.mean(residual ** 2)

    def _horizontal_residual_loss(
        self,
        state: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        v = state[..., 3:6]
        dv_dt = self._finite_difference(v, t)
        z = state[..., 2:3]
        rho = self._compute_density(z)
        Cd = context[..., 2:3]
        S_ref = self._get_param("S_ref", 0.05).to(state.device).view(1, 1, 1)
        mass = state[..., 13:14]

        drag_ax = self._drag_acc_component(v[..., 0:1], rho, Cd, S_ref, mass)
        drag_ay = self._drag_acc_component(v[..., 1:2], rho, Cd, S_ref, mass)

        residual_x = dv_dt[..., 0:1] + drag_ax
        residual_y = dv_dt[..., 1:2] + drag_ay
        return torch.mean(residual_x ** 2 + residual_y ** 2)
    
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
    
    def quaternion_normalization_loss(
        self,
        pred_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalty for quaternion normalization deviation.
        
        Quaternions should have unit norm: ||q|| = 1
        Loss: mean((||q|| - 1)^2)
        
        Args:
            pred_state: [batch, N, 14] or [N, 14]
            
        Returns:
            Scalar loss
        """
        # Ensure batched format
        if pred_state.dim() == 2:
            pred_state = pred_state.unsqueeze(0)
        
        # Extract quaternions: [batch, N, 4] (indices 6:10)
        quaternions = pred_state[:, :, 6:10]  # [batch, N, 4]
        
        # Compute quaternion norm: [batch, N]
        quat_norm = torch.linalg.norm(quaternions, dim=-1)
        
        # Penalty for deviation from 1.0
        norm_deviation = (quat_norm - 1.0) ** 2
        
        return torch.mean(norm_deviation)
    
    def mass_flow_consistency_loss(
        self,
        pred_state: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalty for mass flow violations (mass should be non-increasing).
        
        Mass should decrease or stay constant: m(t+1) <= m(t)
        Loss: mean(ReLU(m(t) - m(t+1))) for all time steps
        
        Args:
            pred_state: [batch, N, 14] or [N, 14]
            t: Time [batch, N, 1] or [N, 1]
            
        Returns:
            Scalar loss
        """
        # Ensure batched format
        if pred_state.dim() == 2:
            pred_state = pred_state.unsqueeze(0)
            t = t.unsqueeze(0)
        
        # Extract mass: [batch, N, 1] (index 13)
        mass = pred_state[:, :, 13:14]  # [batch, N, 1]
        
        # Compute mass differences: m(t) - m(t+1)
        # Should be >= 0 (non-increasing)
        mass_diff = mass[:, :-1, :] - mass[:, 1:, :]  # [batch, N-1, 1]
        
        # Penalty for negative differences (mass increases)
        violations = torch.relu(-mass_diff)  # Only penalize when mass increases
        
        return torch.mean(violations)
    
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
        control: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
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
        
        # Enhanced loss components
        L_quat_norm = self.quaternion_normalization_loss(pred_state)
        L_mass_flow = self.mass_flow_consistency_loss(pred_state, t)

        context_broadcast = None
        if context is not None and (
            self.lambda_mass_residual > 0
            or self.lambda_vz_residual > 0
            or self.lambda_vxy_residual > 0
        ):
            batch_size = pred_state.shape[0] if pred_state.dim() == 3 else 1
            N = pred_state.shape[1] if pred_state.dim() == 3 else pred_state.shape[0]
            context_broadcast = self._broadcast_context(
                context, batch_size, N, pred_state.device
            )

        L_mass_residual = torch.tensor(0.0, device=pred_state.device)
        L_vz_residual = torch.tensor(0.0, device=pred_state.device)
        L_vxy_residual = torch.tensor(0.0, device=pred_state.device)
        L_smooth_z = torch.tensor(0.0, device=pred_state.device)
        L_smooth_vz = torch.tensor(0.0, device=pred_state.device)
        L_pos_vel = torch.tensor(0.0, device=pred_state.device)
        L_smooth_pos = torch.tensor(0.0, device=pred_state.device)
        L_zero_vxy = torch.tensor(0.0, device=pred_state.device)
        L_zero_axy = torch.tensor(0.0, device=pred_state.device)
        L_hacc = torch.tensor(0.0, device=pred_state.device)
        L_xy_zero = torch.tensor(0.0, device=pred_state.device)

        if (
            self.lambda_mass_residual > 0.0
            and context_broadcast is not None
        ):
            L_mass_residual = self._mass_residual_loss(
                pred_state, t, context_broadcast
            )

        if (
            self.lambda_vz_residual > 0.0
            and context_broadcast is not None
        ):
            L_vz_residual = self._vertical_residual_loss(
                pred_state, t, context_broadcast
            )

        if (
            self.lambda_vxy_residual > 0.0
            and context_broadcast is not None
        ):
            L_vxy_residual = self._horizontal_residual_loss(
                pred_state, t, context_broadcast
            )

        if self.lambda_smooth_z > 0.0:
            L_smooth_z = self._second_difference(pred_state[..., 2:3])

        if self.lambda_smooth_vz > 0.0:
            L_smooth_vz = self._second_difference(pred_state[..., 5:6])

        if self.lambda_pos_vel > 0.0:
            L_pos_vel = self._position_velocity_consistency_loss(pred_state, t)

        if self.lambda_smooth_pos > 0.0:
            L_smooth_pos = self._position_smoothing_loss(pred_state)

        if self.lambda_zero_vxy > 0.0:
            L_zero_vxy = self._zero_horizontal_velocity_loss(pred_state, true_state)

        if self.lambda_zero_axy > 0.0:
            L_zero_axy = self._zero_horizontal_acceleration_loss(pred_state, true_state, t)

        if self.lambda_hacc > 0.0:
            L_hacc = self._global_horizontal_acceleration_loss(pred_state, t)

        if self.lambda_xy_zero > 0.0:
            L_xy_zero = self._xy_zero_position_loss(pred_state, true_state)
        
        total_loss = (
            self.lambda_data * L_data
            + self.lambda_phys * L_phys
            + self.lambda_bc * L_bc
            + self.lambda_quat_norm * L_quat_norm
            + self.lambda_mass_flow * L_mass_flow
            + self.lambda_mass_residual * L_mass_residual
            + self.lambda_vz_residual * L_vz_residual
            + self.lambda_vxy_residual * L_vxy_residual
            + self.lambda_smooth_z * L_smooth_z
            + self.lambda_smooth_vz * L_smooth_vz
            + self.lambda_pos_vel * L_pos_vel
            + self.lambda_smooth_pos * L_smooth_pos
            + self.lambda_zero_vxy * L_zero_vxy
            + self.lambda_zero_axy * L_zero_axy
            + self.lambda_hacc * L_hacc
            + self.lambda_xy_zero * L_xy_zero
        )
        
        loss_dict = {
            "total": total_loss,
            "data": L_data,
            "physics": L_phys,
            "boundary": L_bc,
            "quat_norm": L_quat_norm,
            "mass_flow": L_mass_flow,
            "mass_residual": L_mass_residual,
            "vz_residual": L_vz_residual,
            "vxy_residual": L_vxy_residual,
            "smooth_z": L_smooth_z,
            "smooth_vz": L_smooth_vz,
            "pos_vel": L_pos_vel,
            "smooth_pos": L_smooth_pos,
            "zero_vxy": L_zero_vxy.detach(),
            "zero_axy": L_zero_axy.detach(),
            "hacc": L_hacc.detach(),
            "xy_zero": L_xy_zero.detach(),
        }
        
        return total_loss, loss_dict

