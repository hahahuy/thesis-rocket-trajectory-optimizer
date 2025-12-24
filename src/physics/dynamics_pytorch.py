"""
PyTorch implementation of 6-DOF rocket dynamics for autograd.

This module provides differentiable dynamics functions for use in PINN training.
The implementation mirrors the CasADi version but uses PyTorch tensors for autograd.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix (body to inertial).
    
    Args:
        q: Quaternion [..., 4] = [q0, q1, q2, q3] (w, x, y, z)
        
    Returns:
        R: Rotation matrix [..., 3, 3]
    """
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Ensure quaternion is normalized
    q_norm = torch.sqrt(torch.sum(q**2, dim=-1, keepdim=True) + 1e-12)
    q = q / q_norm
    
    R = torch.zeros(*q.shape[:-1], 3, 3, device=q.device, dtype=q.dtype)
    
    R[..., 0, 0] = q0*q0 + q1*q1 - q2*q2 - q3*q3
    R[..., 0, 1] = 2*(q1*q2 - q0*q3)
    R[..., 0, 2] = 2*(q1*q3 + q0*q2)
    
    R[..., 1, 0] = 2*(q1*q2 + q0*q3)
    R[..., 1, 1] = q0*q0 - q1*q1 + q2*q2 - q3*q3
    R[..., 1, 2] = 2*(q2*q3 - q0*q1)
    
    R[..., 2, 0] = 2*(q1*q3 - q0*q2)
    R[..., 2, 1] = 2*(q2*q3 + q0*q1)
    R[..., 2, 2] = q0*q0 - q1*q1 - q2*q2 + q3*q3
    
    return R


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions: q1 * q2.
    
    Args:
        q1: First quaternion [..., 4]
        q2: Second quaternion [..., 4]
        
    Returns:
        q: Product quaternion [..., 4]
    """
    q0_1, q1_1, q2_1, q3_1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    q0_2, q1_2, q2_2, q3_2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    q0 = q0_1*q0_2 - q1_1*q1_2 - q2_1*q2_2 - q3_1*q3_2
    q1 = q0_1*q1_2 + q1_1*q0_2 + q2_1*q3_2 - q3_1*q2_2
    q2 = q0_1*q2_2 - q1_1*q3_2 + q2_1*q0_2 + q3_1*q1_2
    q3 = q0_1*q3_2 + q1_1*q2_2 - q2_1*q1_2 + q3_1*q0_2
    
    return torch.stack([q0, q1, q2, q3], dim=-1)


def compute_dynamics(
    x: torch.Tensor,
    u: torch.Tensor,
    params: Dict[str, torch.Tensor],
    scales: Dict[str, float] = None
) -> torch.Tensor:
    """
    Compute state derivative for 6-DOF rocket dynamics (nondimensional).
    
    Args:
        x: State vector [..., 14] = [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m] (nondim)
        u: Control vector [..., 4] = [T, theta_g, phi_g, delta] (nondim)
        params: Dictionary of physical parameters (nondim)
        scales: Scaling factors for dimensionalization (optional, for dimensional params)
        
    Returns:
        xdot: State derivative [..., 14] (nondim)
    """
    # Unpack state
    r_i = x[..., 0:3]      # Position (nondim)
    v_i = x[..., 3:6]      # Velocity (nondim)
    q = x[..., 6:10]       # Quaternion
    w_b = x[..., 10:13]    # Angular velocity (nondim)
    m = x[..., 13:14]      # Mass (nondim)
    
    # Unpack control
    T = u[..., 0:1]        # Thrust (nondim)
    theta_g = u[..., 1:2]  # Gimbal angle (pitch) [rad]
    phi_g = u[..., 2:3]    # Gimbal angle (yaw) [rad]
    delta = u[..., 3:4]    # Control surface deflection [rad]
    
    # Extract parameters (assumed nondimensional)
    Cd = params.get('Cd', torch.tensor(0.3, device=x.device, dtype=x.dtype))
    CL_alpha = params.get('CL_alpha', torch.tensor(3.5, device=x.device, dtype=x.dtype))
    Cm_alpha = params.get('Cm_alpha', torch.tensor(-0.8, device=x.device, dtype=x.dtype))
    C_delta = params.get('C_delta', torch.tensor(0.05, device=x.device, dtype=x.dtype))
    S_ref = params.get('S_ref', torch.tensor(0.05, device=x.device, dtype=x.dtype))
    l_ref = params.get('l_ref', torch.tensor(1.2, device=x.device, dtype=x.dtype))
    Isp = params.get('Isp', torch.tensor(300.0, device=x.device, dtype=x.dtype))
    g0 = params.get('g0', torch.tensor(9.81, device=x.device, dtype=x.dtype))
    rho0 = params.get('rho0', torch.tensor(1.225, device=x.device, dtype=x.dtype))
    h_scale = params.get('h_scale', torch.tensor(8400.0, device=x.device, dtype=x.dtype))
    
    # Inertia tensor (assumed diagonal, nondim)
    I_b = params.get('I_b', torch.tensor([1000.0, 1000.0, 100.0], device=x.device, dtype=x.dtype))
    if I_b.dim() == 1 and len(I_b) == 3:
        I_xx, I_yy, I_zz = I_b[0], I_b[1], I_b[2]
    else:
        # Full matrix case (simplified for now - extract diagonal)
        if I_b.dim() == 2:
            I_xx, I_yy, I_zz = I_b[0, 0], I_b[1, 1], I_b[2, 2]
        else:
            I_xx, I_yy, I_zz = I_b[0], I_b[1], I_b[2]
    
    # Limits (nondim)
    T_max = params.get('T_max', torch.tensor(10.0, device=x.device, dtype=x.dtype))
    m_dry = params.get('m_dry', torch.tensor(0.2, device=x.device, dtype=x.dtype))
    
    # Clamp thrust and mass
    T = torch.clamp(T, min=0.0, max=T_max)
    m = torch.clamp(m, min=m_dry)
    
    # Atmospheric density (exponential model)
    # If scales provided, dimensionalize altitude for density calculation
    if scales is not None:
        altitude = r_i[..., 2:3] * scales['L']
        rho_dim = rho0 * torch.exp(-torch.clamp(altitude, min=0.0) / h_scale)
        rho = rho_dim / scales.get('RHO', 1.225)  # Nondimensionalize
    else:
        # Assume r_i is already in meters (for density calc)
        altitude = r_i[..., 2:3]
        rho = rho0 * torch.exp(-torch.clamp(altitude, min=0.0) / h_scale)
    
    # Wind (simplified: no wind for now)
    wind_i = torch.zeros_like(v_i)
    v_rel_i = v_i - wind_i
    
    # Use smooth norm
    v_rel_norm_smooth = torch.sqrt(torch.sum(v_rel_i**2, dim=-1, keepdim=True) + 1e-12)
    
    # Rotation matrix from quaternion (body to inertial)
    q_norm_smooth = torch.sqrt(torch.sum(q**2, dim=-1, keepdim=True) + 1e-12)
    q_normalized = q / q_norm_smooth
    
    # Determine if batched (input is 2D: [batch, 4] or [batch*N, 4])
    # Always treat as batched if q is 2D (even if batch_size=1)
    is_batched = q.dim() == 2
    
    # Compute rotation matrices
    if is_batched:
        # Batched: [batch, 4] or [batch*N, 4]
        batch_size = q.shape[0]
        R_b2i = torch.stack([quaternion_to_rotation_matrix(q_normalized[i]) for i in range(batch_size)])
        # R_b2i: [batch, 3, 3]
    else:
        # Unbatched: [4] - single sample
        R_b2i = quaternion_to_rotation_matrix(q_normalized)
        R_b2i = R_b2i.unsqueeze(0)  # [1, 3, 3] to make it batched
        is_batched = True  # Treat as batched for consistency
    
    # Relative velocity in body frame
    if is_batched and R_b2i.dim() == 3:
        # Batched: [batch, 3, 3] and v_rel_i: [batch, 3]
        v_rel_i_expanded = v_rel_i.unsqueeze(-1)  # [batch, 3, 1]
        v_rel_b = torch.bmm(R_b2i.transpose(-2, -1), v_rel_i_expanded).squeeze(-1)  # [batch, 3]
    else:
        # Unbatched: [3, 3] and [3]
        R_b2i_single = R_b2i.squeeze(0) if R_b2i.dim() == 3 else R_b2i
        v_rel_i_single = v_rel_i.squeeze(0) if v_rel_i.dim() == 2 else v_rel_i
        v_rel_b = (R_b2i_single.T @ v_rel_i_single.unsqueeze(-1)).squeeze(-1)
        if v_rel_i.dim() == 2:
            v_rel_b = v_rel_b.unsqueeze(0)
    
    # Dynamic pressure (nondim)
    # q_dyn = 0.5 * rho * v^2, but need to handle scaling
    if scales is not None:
        v_dim = v_rel_norm_smooth * scales['V']
        rho_dim = rho * scales.get('RHO', 1.225)
        q_dyn_dim = 0.5 * rho_dim * v_dim**2
        q_dyn = q_dyn_dim / scales.get('Q', 1e4)  # Nondimensionalize
    else:
        q_dyn = 0.5 * rho * v_rel_norm_smooth**2
    
    # Angle of attack
    v_rel_b_x_smooth = torch.sqrt(v_rel_b[..., 0:1]**2 + 1e-12) * torch.sign(v_rel_b[..., 0:1] + 1e-15)
    alpha = torch.atan2(v_rel_b[..., 2:3], v_rel_b_x_smooth)
    
    # Thrust direction unit vector in body frame
    # Use cat instead of stack to avoid dimension issues
    uT_b_x = torch.cos(theta_g) * torch.cos(phi_g)
    uT_b_y = torch.sin(phi_g)
    uT_b_z = torch.sin(theta_g) * torch.cos(phi_g)
    uT_b = torch.cat([uT_b_x, uT_b_y, uT_b_z], dim=-1)  # [batch, 3]
    
    # Normalize
    uT_b_norm = torch.sqrt(torch.sum(uT_b**2, dim=-1, keepdim=True) + 1e-12)
    uT_b = uT_b / uT_b_norm
    
    # Thrust force in body frame (nondim)
    # Ensure T and uT_b have compatible shapes
    if T.dim() == 1:
        T = T.unsqueeze(-1)
    F_T_b = T * uT_b  # [batch, 3] or [batch*N, 3]
    
    # Drag force in body frame
    v_rel_b_norm = torch.sqrt(torch.sum(v_rel_b**2, dim=-1, keepdim=True) + 1e-12)
    u_drag_b = -v_rel_b / v_rel_b_norm
    # Ensure q_dyn has right shape for broadcasting
    if q_dyn.dim() == 1:
        q_dyn = q_dyn.unsqueeze(-1)
    F_D_b = -q_dyn * S_ref * Cd * u_drag_b
    
    # Lift force in body frame
    v_rel_b_x_smooth = torch.sqrt(v_rel_b[..., 0:1]**2 + 1e-12)
    # Stack along last dimension to get [batch, 3]
    e_lift_b_x = -v_rel_b[..., 2:3] / (v_rel_b_x_smooth + 1e-12)
    e_lift_b_y = torch.zeros_like(v_rel_b[..., 0:1])
    e_lift_b_z = v_rel_b[..., 0:1] / (v_rel_b_x_smooth + 1e-12)
    e_lift_b = torch.cat([e_lift_b_x, e_lift_b_y, e_lift_b_z], dim=-1)  # [batch, 3]
    e_lift_b_norm = torch.sqrt(torch.sum(e_lift_b**2, dim=-1, keepdim=True) + 1e-12)
    e_lift_b = e_lift_b / e_lift_b_norm
    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(-1)
    F_L_b = q_dyn * S_ref * CL_alpha * alpha * e_lift_b
    
    # Total force in body frame
    F_b = F_T_b + F_D_b + F_L_b  # Should be [batch, 3] or [batch*N, 3]
    
    # Ensure F_b is 2D: [batch, 3]
    if F_b.dim() == 1:
        F_b = F_b.unsqueeze(0)
    elif F_b.dim() == 3:
        # If somehow 3D, reshape
        F_b = F_b.view(-1, 3)
    
    # Transform to inertial frame
    if is_batched and R_b2i.dim() == 3:
        # Batched: R_b2i [batch, 3, 3], F_b [batch, 3]
        # Ensure shapes match
        if F_b.shape[0] != R_b2i.shape[0]:
            # Reshape if needed
            F_b = F_b.view(R_b2i.shape[0], -1)
        F_b_expanded = F_b.unsqueeze(-1)  # [batch, 3, 1]
        F_i = torch.bmm(R_b2i, F_b_expanded).squeeze(-1)  # [batch, 3]
    else:
        # Unbatched
        R_b2i_single = R_b2i.squeeze(0) if R_b2i.dim() == 3 else R_b2i
        F_b_single = F_b.squeeze(0) if F_b.dim() == 2 else F_b
        F_i = (R_b2i_single @ F_b_single.unsqueeze(-1)).squeeze(-1)
        if F_b.dim() == 2:
            F_i = F_i.unsqueeze(0)
    
    # Gravity (nondim)
    if scales is not None:
        g_dim = torch.tensor([0.0, 0.0, -g0], device=x.device, dtype=x.dtype)
        g_scale = scales['V']**2 / scales['L']
        g_i_scalar = g_dim / g_scale  # [3] - nondimensionalize acceleration
        # Broadcast to match r_i shape: [batch*N, 3]
        if is_batched:
            batch_size_flat = r_i.shape[0]
            if batch_size_flat > 0:
                g_i = g_i_scalar.unsqueeze(0).repeat(batch_size_flat, 1)  # [batch*N, 3]
            else:
                g_i = g_i_scalar.unsqueeze(0)  # [1, 3] fallback
        else:
            g_i = g_i_scalar  # [3]
    else:
        g_i = torch.zeros_like(r_i)
        g_i[..., 2] = -g0 / (313.0**2 / 10000.0)  # Approximate scaling
    
    # Position derivative
    r_dot = v_i
    
    # Velocity derivative
    m_safe = torch.clamp(m, min=1e-3)
    # Ensure shapes are compatible for broadcasting
    # F_i: [batch*N, 3], m_safe: [batch*N, 1], g_i: [batch*N, 3] or broadcastable
    # Ensure m_safe has shape [batch*N, 1]
    if m_safe.dim() == 1:
        m_safe = m_safe.unsqueeze(-1)
    # Ensure g_i matches F_i shape for broadcasting
    if g_i.dim() == 1 and g_i.shape[0] == 3:
        g_i = g_i.unsqueeze(0).expand(F_i.shape[0], -1)
    elif g_i.dim() == 2:
        if g_i.shape[0] == 1 and g_i.shape[1] == 3:
            g_i = g_i.expand(F_i.shape[0], -1)
        elif g_i.shape[0] != F_i.shape[0] and g_i.shape[1] == 3:
            g_i = g_i[0:1].expand(F_i.shape[0], -1)
    v_dot = F_i / m_safe + g_i
    
    # Quaternion derivative: q_dot = 0.5 * q * [0, w]
    omega_quat = torch.cat([
        torch.zeros_like(w_b[..., 0:1]),
        w_b
    ], dim=-1)
    
    if q.dim() == 2:
        # Batched
        q_dot = torch.stack([
            quaternion_multiply(q_normalized[i], omega_quat[i]) for i in range(q.shape[0])
        ]) * 0.5
    else:
        # Unbatched
        q_dot = quaternion_multiply(q_normalized, omega_quat) * 0.5
    
    # Angular velocity derivative: w_dot = I^(-1) * (M - w × I*w)
    # Moments in body frame
    M_aero_b = torch.zeros_like(w_b)
    M_aero_b[..., 1:2] = q_dyn * S_ref * l_ref * (Cm_alpha * alpha + C_delta * delta)
    
    # Thrust moment (simplified: assume no offset)
    M_T_b = torch.zeros_like(w_b)
    M_b = M_aero_b + M_T_b
    
    # Coriolis term: w × (I*w)
    I_w = torch.stack([
        I_xx * w_b[..., 0],
        I_yy * w_b[..., 1],
        I_zz * w_b[..., 2]
    ], dim=-1)
    w_cross_Iw = torch.cross(w_b, I_w, dim=-1)
    
    # Angular acceleration
    net_moment = M_b - w_cross_Iw
    w_dot = torch.stack([
        net_moment[..., 0] / torch.clamp(I_xx, min=1e-3),
        net_moment[..., 1] / torch.clamp(I_yy, min=1e-3),
        net_moment[..., 2] / torch.clamp(I_zz, min=1e-3)
    ], dim=-1)
    
    # Mass derivative (nondim)
    T_safe = torch.clamp(T, min=0.0)
    if scales is not None:
        # m_dot = -T / (Isp * g0) in dimensional, then nondimensionalize
        m_dot_dim = -T_safe * scales['F'] / (Isp * g0)
        m_dot = m_dot_dim / (scales['M'] / scales['T'])
    else:
        # Approximate scaling
        m_dot = -T_safe / (Isp * g0 / (50.0 / 31.62))
    
    # State derivative
    xdot = torch.cat([
        r_dot,      # [0:3]
        v_dot,      # [3:6]
        q_dot,      # [6:10]
        w_dot,      # [10:13]
        m_dot       # [13]
    ], dim=-1)
    
    return xdot


class DynamicsModule(nn.Module):
    """
    PyTorch module wrapper for dynamics computation.
    Useful for integrating into larger models.
    """
    
    def __init__(self, params: Dict[str, float], scales: Dict[str, float] = None):
        super().__init__()
        self.params = {k: torch.tensor(v, dtype=torch.float32) for k, v in params.items()}
        self.scales = scales or {}
        
    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute state derivative.
        
        Args:
            x: State [..., 14] (nondim)
            u: Control [..., 4] (nondim)
            
        Returns:
            xdot: State derivative [..., 14] (nondim)
        """
        # Move params to same device as x
        params_device = {k: v.to(x.device) for k, v in self.params.items()}
        return compute_dynamics(x, u, params_device, self.scales)

