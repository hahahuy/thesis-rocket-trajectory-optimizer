"""
CasADi-based 6-DOF rocket dynamics for optimal control.

This module implements the dynamics equations in CasADi symbolic form
for use in direct collocation transcription.
"""

import casadi as ca
import numpy as np
from typing import Tuple, Dict


def compute_dynamics(
    x: ca.MX, u: ca.MX, params: Dict
) -> ca.MX:
    """
    Compute state derivative for 6-DOF rocket dynamics.
    
    Args:
        x: State vector [14] = [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]
        u: Control vector [4] = [T, theta_g, phi_g, delta]
        params: Dictionary of physical parameters
        
    Returns:
        xdot: State derivative [14]
    """
    # Unpack state
    r_i = x[0:3]      # Position [m]
    v_i = x[3:6]      # Velocity [m/s]
    q = x[6:10]       # Quaternion
    w_b = x[10:13]    # Angular velocity [rad/s]
    m = x[13]         # Mass [kg]
    
    # Unpack control
    T = u[0]          # Thrust [N]
    theta_g = u[1]    # Gimbal angle (pitch) [rad]
    phi_g = u[2]      # Gimbal angle (yaw) [rad]
    delta = u[3]      # Control surface deflection [rad]
    
    # Physical parameters
    Cd = params['Cd']
    CL_alpha = params['CL_alpha']
    Cm_alpha = params['Cm_alpha']
    C_delta = params['C_delta']
    S_ref = params['S_ref']
    l_ref = params['l_ref']
    Isp = params['Isp']
    g0 = params['g0']
    rho0 = params['rho0']
    h_scale = params['h_scale']
    
    # Inertia tensor (assumed diagonal)
    I_b = params['I_b']  # Should be 3x3 matrix or diagonal vector
    
    # Limits
    T_max = params['T_max']
    m_dry = params['m_dry']
    
    # Clamp thrust and mass
    T = ca.fmin(T, T_max)
    m = ca.fmax(m, m_dry)
    
    # Atmospheric density (exponential model)
    altitude = r_i[2]  # z-component
    rho = rho0 * ca.exp(-ca.fmax(altitude, 0.0) / h_scale)
    
    # Wind (simplified: no wind for now, can be added)
    wind_i = ca.MX.zeros(3)
    v_rel_i = v_i - wind_i
    # Use smooth norm to avoid AD issues: sqrt(v'v + eps) instead of fmax(norm(v), eps)
    v_rel_norm_smooth = ca.sqrt(ca.dot(v_rel_i, v_rel_i) + 1e-12)
    v_rel_norm_safe = v_rel_norm_smooth
    
    # Rotation matrix from quaternion (body to inertial)
    # Use smooth normalization to avoid AD issues with fmax
    # q_normalized = q / sqrt(q'q + eps) instead of q / fmax(norm(q), eps)
    q_norm_smooth = ca.sqrt(ca.dot(q, q) + 1e-12)
    q_normalized = q / q_norm_smooth
    R_b2i = quaternion_to_rotation_matrix(q_normalized)
    
    # Relative velocity in body frame
    # Ensure rotation matrix is valid (quaternion should be normalized)
    v_rel_b = R_b2i.T @ v_rel_i
    
    # Dynamic pressure
    q_dyn = 0.5 * rho * v_rel_norm_safe ** 2
    
    # Angle of attack (use smooth approximation to avoid AD issues)
    v_rel_b_x_smooth = ca.sqrt(v_rel_b[0]**2 + 1e-12) * ca.sign(v_rel_b[0] + 1e-15)
    alpha = ca.atan2(v_rel_b[2], v_rel_b_x_smooth)
    
    # Thrust direction unit vector in body frame (from gimbal angles)
    uT_b = ca.vertcat(
        ca.cos(theta_g) * ca.cos(phi_g),
        ca.sin(phi_g),
        ca.sin(theta_g) * ca.cos(phi_g)
    )
    
    # Normalize to ensure unit vector (use smooth normalization for AD)
    uT_b_norm_smooth = ca.sqrt(ca.dot(uT_b, uT_b) + 1e-12)
    uT_b = uT_b / uT_b_norm_smooth
    
    # Thrust force in body frame
    F_T_b = T * uT_b
    
    # Drag force in body frame (opposite to relative velocity)
    # Use smooth normalization to avoid AD issues: u = -v / (norm(v) + eps)
    # This avoids division by fmax which can cause AD problems
    v_rel_b_norm_smooth = ca.sqrt(ca.dot(v_rel_b, v_rel_b) + 1e-12)
    u_drag_b = -v_rel_b / v_rel_b_norm_smooth
    F_D_b = -q_dyn * S_ref * Cd * u_drag_b
    
    # Lift force in body frame (perpendicular to velocity, in x-z plane)
    # Use smooth approximation for division to avoid AD issues
    v_rel_b_x_smooth = ca.sqrt(v_rel_b[0]**2 + 1e-12)
    e_lift_b = ca.vertcat(
        -v_rel_b[2] / v_rel_b_x_smooth,
        0.0,
        v_rel_b[0] / v_rel_b_x_smooth
    )
    # Smooth normalization for lift direction
    e_lift_b_norm_smooth = ca.sqrt(ca.dot(e_lift_b, e_lift_b) + 1e-12)
    e_lift_b = e_lift_b / e_lift_b_norm_smooth
    F_L_b = q_dyn * S_ref * CL_alpha * alpha * e_lift_b
    
    # Total force in body frame
    F_b = F_T_b + F_D_b + F_L_b
    
    # Transform to inertial frame
    F_i = R_b2i @ F_b
    
    # Gravity
    g_i = ca.vertcat(0.0, 0.0, -g0)
    
    # Position derivative
    r_dot = v_i
    
    # Velocity derivative (ensure m is not zero or negative)
    m_safe = ca.fmax(m, 1e-3)  # Prevent division by zero
    v_dot = F_i / m_safe + g_i
    
    # Quaternion derivative: q_dot = 0.5 * q * [0, w]
    omega_quat = ca.vertcat(0.0, w_b[0], w_b[1], w_b[2])
    q_dot = 0.5 * quaternion_multiply(q, omega_quat)
    
    # Angular velocity derivative: w_dot = I^(-1) * (M - w × I*w)
    # Moments in body frame
    M_aero_b = ca.vertcat(
        0.0,
        q_dyn * S_ref * l_ref * (Cm_alpha * alpha + C_delta * delta),
        0.0
    )
    
    # Thrust moment (gimbal offset, simplified: assume no offset for now)
    M_T_b = ca.MX.zeros(3)
    
    # Total moment
    M_b = M_aero_b + M_T_b
    
    # Inertia tensor (assume diagonal)
    I_b_arr = np.array(I_b, dtype=float).flatten()
    
    if len(I_b_arr) == 3:
        # Diagonal inertia - element-wise multiplication for I*w
        # For diagonal matrix: I*w = [Ixx*wx, Iyy*wy, Izz*wz]
        I_w = ca.vertcat(
            I_b_arr[0] * w_b[0],
            I_b_arr[1] * w_b[1],
            I_b_arr[2] * w_b[2]
        )
    elif len(I_b_arr) == 9:
        # Full 3x3 matrix stored as flat list
        I_b_mat = ca.DM(I_b_arr.reshape(3, 3))
        I_w = I_b_mat @ w_b
        I_inv = ca.inv(I_b_mat)
    else:
        # Default: assume diagonal
        I_b_arr = np.array([I_b_arr[0] if len(I_b_arr) > 0 else 1000.0,
                           I_b_arr[4] if len(I_b_arr) > 4 else 1000.0,
                           I_b_arr[8] if len(I_b_arr) > 8 else 100.0])
        I_w = ca.vertcat(
            I_b_arr[0] * w_b[0],
            I_b_arr[1] * w_b[1],
            I_b_arr[2] * w_b[2]
        )
    
    # Coriolis term: w × (I*w)
    w_cross_Iw = ca.cross(w_b, I_w)
    
    # Angular acceleration
    I_b_arr = np.array(I_b, dtype=float).flatten()
    if len(I_b_arr) == 3:
        # For diagonal inertia, compute element-wise
        # Ensure I values are positive to avoid division by zero
        I_xx = ca.fmax(float(I_b_arr[0]), 1e-3)
        I_yy = ca.fmax(float(I_b_arr[1]), 1e-3)
        I_zz = ca.fmax(float(I_b_arr[2]), 1e-3)
        net_moment = M_b - w_cross_Iw
        w_dot = ca.vertcat(
            net_moment[0] / I_xx,
            net_moment[1] / I_yy,
            net_moment[2] / I_zz
        )
    elif len(I_b_arr) == 9:
        # Full matrix - already computed I_inv above
        w_dot = I_inv @ (M_b - w_cross_Iw)
    else:
        # Default diagonal with safe values
        I_xx = ca.fmax(float(I_b_arr[0]) if len(I_b_arr) > 0 else 1000.0, 1e-3)
        I_yy = ca.fmax(float(I_b_arr[4]) if len(I_b_arr) > 4 else 1000.0, 1e-3)
        I_zz = ca.fmax(float(I_b_arr[8]) if len(I_b_arr) > 8 else 100.0, 1e-3)
        net_moment = M_b - w_cross_Iw
        w_dot = ca.vertcat(
            net_moment[0] / I_xx,
            net_moment[1] / I_yy,
            net_moment[2] / I_zz
        )
    
    # Mass derivative (clamp to prevent negative mass)
    T_safe = ca.fmax(T, 0.0)
    m_dot = -T_safe / (Isp * g0)
    
    # State derivative
    xdot = ca.vertcat(
        r_dot,      # [0:3]
        v_dot,      # [3:6]
        q_dot,      # [6:10]
        w_dot,      # [10:13]
        m_dot       # [13]
    )
    
    return xdot


def quaternion_to_rotation_matrix(q: ca.MX) -> ca.MX:
    """
    Convert quaternion to rotation matrix (body to inertial).
    
    Args:
        q: Quaternion [q0, q1, q2, q3] (w, x, y, z)
        
    Returns:
        R: 3x3 rotation matrix
    """
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    
    R = ca.MX.zeros(3, 3)
    
    R[0, 0] = q0*q0 + q1*q1 - q2*q2 - q3*q3
    R[0, 1] = 2*(q1*q2 - q0*q3)
    R[0, 2] = 2*(q1*q3 + q0*q2)
    
    R[1, 0] = 2*(q1*q2 + q0*q3)
    R[1, 1] = q0*q0 - q1*q1 + q2*q2 - q3*q3
    R[1, 2] = 2*(q2*q3 - q0*q1)
    
    R[2, 0] = 2*(q1*q3 - q0*q2)
    R[2, 1] = 2*(q2*q3 + q0*q1)
    R[2, 2] = q0*q0 - q1*q1 - q2*q2 + q3*q3
    
    return R


def quaternion_multiply(q1: ca.MX, q2: ca.MX) -> ca.MX:
    """
    Multiply two quaternions: q1 * q2.
    
    Args:
        q1: First quaternion [q0, q1, q2, q3]
        q2: Second quaternion [q0, q1, q2, q3]
        
    Returns:
        q: Product quaternion
    """
    q0_1, q1_1, q2_1, q3_1 = q1[0], q1[1], q1[2], q1[3]
    q0_2, q1_2, q2_2, q3_2 = q2[0], q2[1], q2[2], q2[3]
    
    q0 = q0_1*q0_2 - q1_1*q1_2 - q2_1*q2_2 - q3_1*q3_2
    q1 = q0_1*q1_2 + q1_1*q0_2 + q2_1*q3_2 - q3_1*q2_2
    q2 = q0_1*q2_2 - q1_1*q3_2 + q2_1*q0_2 + q3_1*q1_2
    q3 = q0_1*q3_2 + q1_1*q2_2 - q2_1*q1_2 + q3_1*q0_2
    
    return ca.vertcat(q0, q1, q2, q3)


def compute_dynamic_pressure(x: ca.MX, params: Dict) -> ca.MX:
    """
    Compute dynamic pressure.
    
    Args:
        x: State vector [14]
        params: Physical parameters
        
    Returns:
        q: Dynamic pressure [Pa]
    """
    r_i = x[0:3]
    v_i = x[3:6]
    
    rho0 = params['rho0']
    h_scale = params['h_scale']
    
    altitude = r_i[2]
    rho = rho0 * ca.exp(-ca.fmax(altitude, 0.0) / h_scale)
    
    v_rel_i = v_i  # Assuming no wind
    # Use smooth norm to avoid AD issues
    v_rel_norm_smooth = ca.sqrt(ca.dot(v_rel_i, v_rel_i) + 1e-12)
    v_rel_norm_safe = v_rel_norm_smooth
    
    q = 0.5 * rho * v_rel_norm_safe ** 2
    return q


def compute_load_factor(x: ca.MX, u: ca.MX, params: Dict) -> ca.MX:
    """
    Compute load factor.
    
    Args:
        x: State vector [14]
        u: Control vector [4]
        params: Physical parameters
        
    Returns:
        n: Load factor [g]
    """
    r_i = x[0:3]
    v_i = x[3:6]
    q = x[6:10]
    m = x[13]
    
    g0 = params['g0']
    rho0 = params['rho0']
    h_scale = params['h_scale']
    S_ref = params['S_ref']
    CL_alpha = params['CL_alpha']
    
    # Atmospheric density
    altitude = r_i[2]
    rho = rho0 * ca.exp(-ca.fmax(altitude, 0.0) / h_scale)
    
    # Relative velocity
    v_rel_i = v_i
    # Use smooth norm to avoid AD issues
    v_rel_norm_smooth = ca.sqrt(ca.dot(v_rel_i, v_rel_i) + 1e-12)
    v_rel_norm_safe = v_rel_norm_smooth
    
    # Dynamic pressure
    q_dyn = 0.5 * rho * v_rel_norm_safe ** 2
    
    # Rotation matrix (ensure quaternion is normalized)
    # Use smooth normalization to avoid AD issues
    q_norm_smooth = ca.sqrt(ca.dot(q, q) + 1e-12)
    q_safe = q / q_norm_smooth
    R_b2i = quaternion_to_rotation_matrix(q_safe)
    v_rel_b = R_b2i.T @ v_rel_i
    
    # Angle of attack (use smooth approximation to avoid AD issues)
    v_rel_b_x_smooth = ca.sqrt(v_rel_b[0]**2 + 1e-12) * ca.sign(v_rel_b[0] + 1e-15)
    alpha = ca.atan2(v_rel_b[2], v_rel_b_x_smooth)
    
    # Lift force magnitude
    L_mag = q_dyn * S_ref * CL_alpha * ca.fabs(alpha)
    
    # Load factor (ensure m is positive to avoid division by zero)
    m_safe = ca.fmax(m, 1e-3)  # Prevent division by zero
    # Clamp L_mag to prevent overflow
    L_mag_safe = ca.fmin(L_mag, 1e8)  # Reasonable upper bound
    n = L_mag_safe / (m_safe * g0) + 1.0
    
    return n

