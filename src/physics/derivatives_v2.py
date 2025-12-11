# ======================================
# Central Difference Derivative (v2)
# ======================================

import torch


def central_difference(state, t, eps=1e-12):
    """
    Computes central difference derivative:
        dstate/dt = (s(t+1) - s(t-1)) / (2*dt)
    
    For interior points, uses central difference. For boundaries, falls back
    to forward/backward difference.
    
    Args:
        state: [batch, N, dim] - state trajectory
        t: [batch, N, 1] or [batch, N] - time grid (handles non-uniform grids)
        eps: small epsilon to prevent division by zero
    
    Returns:
        derivative: [batch, N, dim] - time derivative of state
    """
    # Prepare output with same shape
    deriv = torch.zeros_like(state)
    
    batch, N, dim = state.shape
    
    if N < 2:
        return deriv
    
    # Extract time values
    if t.dim() == 3:
        t_vals = t[:, :, 0]  # [batch, N]
    else:
        t_vals = t  # [batch, N]
    
    # Work only on interior points (indices 1 to N-2)
    # Central difference: (s[i+1] - s[i-1]) / (2*dt_central)
    if N >= 3:
        s_next = state[:, 2:, :]  # [batch, N-2, dim]
        s_prev = state[:, :-2, :]  # [batch, N-2, dim]
        
        # Compute dt_central = t[i+1] - t[i-1] for each interior point
        t_next = t_vals[:, 2:]  # [batch, N-2]
        t_prev = t_vals[:, :-2]  # [batch, N-2]
        dt_central = (t_next - t_prev).clamp_min(eps).unsqueeze(-1)  # [batch, N-2, 1]
        
        d_mid = (s_next - s_prev) / (2.0 * dt_central)
        
        # Assign to interior indices (1 to N-2)
        deriv[:, 1:-1, :] = d_mid
    
    # Boundary handling (fallback to forward/backward diff)
    # First point: forward difference
    if N > 1:
        dt_forward = (t_vals[:, 1] - t_vals[:, 0]).clamp_min(eps).unsqueeze(-1)  # [batch, 1]
        deriv[:, 0, :] = (state[:, 1, :] - state[:, 0, :]) / dt_forward
    
    # Last point: backward difference
    if N > 1:
        dt_backward = (t_vals[:, -1] - t_vals[:, -2]).clamp_min(eps).unsqueeze(-1)  # [batch, 1]
        deriv[:, -1, :] = (state[:, -1, :] - state[:, -2, :]) / dt_backward
    
    return deriv

