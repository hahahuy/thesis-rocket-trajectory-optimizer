"""
Direct collocation transcription for optimal control.

Implements Hermite-Simpson collocation method for converting continuous
OCP to discrete NLP.
"""

import casadi as ca
import numpy as np
from typing import Callable, Tuple, Dict


def hermite_simpson_collocation(
    f: Callable,
    x: ca.MX,
    u: ca.MX,
    dt: ca.MX,
    params: Dict
) -> Tuple[ca.MX, ca.MX]:
    """
    Hermite-Simpson collocation step.
    
    For interval [t_k, t_{k+1}]:
    - x_k, x_{k+1}: node states
    - u_k, u_{k+1}: node controls
    - x_m: midpoint state (collocation point)
    - u_m: midpoint control (average)
    
    The defect constraint enforces:
    x_{k+1} = x_k + (h/6) * [f_k + 4*f_m + f_{k+1}]
    
    where:
    x_m = 0.5*(x_k + x_{k+1}) + (h/8)*(f_k - f_{k+1})
    u_m = 0.5*(u_k + u_{k+1})
    
    Args:
        f: Dynamics function f(x, u, params) -> xdot
        x: State at node k (nx,)
        u: Control at node k (nu,)
        dt: Time step h (scalar)
        params: Physical parameters
        
    Returns:
        defect: Defect constraint (nx,) - should be zero
        x_next: Predicted next state (nx,)
    """
    nx = x.size1()
    nu = u.size2() if u.size2() > 0 else u.size1()
    
    # Evaluate dynamics at node k
    f_k = f(x, u, params)
    
    # For Hermite-Simpson, we need x_{k+1} and u_{k+1}
    # In practice, this will be called with x_{k+1} and u_{k+1} passed separately
    # This is a helper function - the actual transcription is in transcription.py
    
    # For now, return a placeholder that will be used in transcription
    x_next = x + dt * f_k  # Euler step (will be replaced by Hermite-Simpson)
    defect = ca.MX.zeros(nx, 1)
    
    return defect, x_next


def compute_hermite_simpson_step(
    f: Callable,
    x_k: ca.MX,
    u_k: ca.MX,
    x_kp1: ca.MX,
    u_kp1: ca.MX,
    dt: ca.MX,
    params: Dict
) -> ca.MX:
    """
    Compute Hermite-Simpson defect constraint.
    
    Args:
        f: Dynamics function
        x_k: State at node k
        u_k: Control at node k
        x_kp1: State at node k+1
        u_kp1: Control at node k+1
        dt: Time step h
        params: Physical parameters
        
    Returns:
        defect: Defect constraint (should be zero)
    """
    # Dynamics at nodes
    f_k = f(x_k, u_k, params)
    f_kp1 = f(x_kp1, u_kp1, params)
    
    # Midpoint state
    x_m = 0.5 * (x_k + x_kp1) + (dt / 8.0) * (f_k - f_kp1)
    
    # Extract midpoint state components before modification
    r_m = x_m[0:3]
    v_m = x_m[3:6]
    q_m = x_m[6:10]
    w_m = x_m[10:13]
    m_m = x_m[13]
    
    # Normalize quaternion and clamp mass (before reconstructing)
    q_m_norm = ca.norm_2(q_m)
    # Use fmax to ensure denominator is never zero (more AD-friendly than if_else)
    q_m_safe = q_m / ca.fmax(q_m_norm, 1e-6)
    # Clamp mass to positive value
    m_m_safe = ca.fmax(m_m, 1e-3)
    
    # Reconstruct midpoint state with safe values
    x_m = ca.vertcat(r_m, v_m, q_m_safe, w_m, m_m_safe)
    
    # Midpoint control (average)
    u_m = 0.5 * (u_k + u_kp1)
    
    # Clamp control to reasonable values
    u_m = ca.vertcat(
        ca.fmax(ca.fmin(u_m[0], 1e6), 0.0),  # Thrust
        ca.fmax(ca.fmin(u_m[1], ca.pi), -ca.pi),  # Gimbal
        ca.fmax(ca.fmin(u_m[2], ca.pi), -ca.pi),  # Gimbal
        ca.fmax(ca.fmin(u_m[3], ca.pi), -ca.pi)   # Control surface
    )
    
    # Dynamics at midpoint
    f_m = f(x_m, u_m, params)
    
    # Hermite-Simpson integration
    x_kp1_predicted = x_k + (dt / 6.0) * (f_k + 4.0 * f_m + f_kp1)
    
    # Defect constraint
    defect = x_kp1 - x_kp1_predicted
    
    # Ensure column vector
    if defect.shape[1] > 1:
        defect = ca.reshape(defect, -1, 1)
    elif defect.shape[0] > 0 and defect.shape[1] == 0:
        defect = ca.reshape(defect, -1, 1)
    
    return defect


def compute_collocation_defects(
    f: Callable,
    X: ca.MX,
    U: ca.MX,
    dt: ca.MX,
    params: Dict
) -> ca.MX:
    """
    Compute all collocation defect constraints.
    
    Args:
        f: Dynamics function
        X: State matrix (nx, N+1) - states at all nodes
        U: Control matrix (nu, N) - controls at all intervals
        dt: Time step (scalar)
        params: Physical parameters
        
    Returns:
        defects: Defect constraints (nx, N) - should all be zero
    """
    nx = X.size1()
    N = U.size2() if U.size2() > 0 else U.size1() // (U.size1() // nx)  # Number of intervals
    
    if X.size2() < N + 1:
        N = X.size2() - 1
    
    defects = []
    
    for k in range(N):
        x_k = X[:, k]
        x_kp1 = X[:, k + 1]
        u_k = U[:, k]
        u_kp1 = U[:, k] if k == N - 1 else U[:, k + 1]
        
        # For last interval, use same control
        if k == N - 1:
            u_kp1 = u_k
        
        defect = compute_hermite_simpson_step(f, x_k, u_k, x_kp1, u_kp1, dt, params)
        defects.append(defect)
    
    return ca.horzcat(*defects) if defects else ca.MX.zeros(nx, N)

