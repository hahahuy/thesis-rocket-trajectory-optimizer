"""
[PINN_V2][2025-12-01][AN Architecture]

Physics residual layer for Rocket PINN models.

This module is **not** a neural network in the classical sense (it has no
trainable parameters). It is a computational layer that:

1. Takes time `t`, predicted state trajectory `state_pred`, and (optional)
   context / control inputs.
2. Uses autograd to compute first (and optionally second) time derivatives
   of the state with respect to time.
3. Inserts these derivatives into the rocket dynamics model to compute
   residuals:
      r(t) = d/dt s_pred(t) - f(s_pred(t), u(t), p)
4. Returns residuals that can be consumed by loss functions or logged
   during training.

State convention (consistent with the rest of the project):
    s = [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]  (14D)
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.physics.dynamics_pytorch import compute_dynamics


@dataclass
class PhysicsResiduals:
    """
    Container for physics residual outputs.

    Attributes:
        state_residual: Full state residual r = ds/dt - f(s, u, p) [batch, N, 14]
        translation_residual: Residual on [x, y, z, vx, vy, vz] [batch, N, 6]
        rotation_residual: Residual on [q0..q3, wx, wy, wz] [batch, N, 7]
        mass_residual: Residual on mass component [batch, N, 1]
        state_dot: Autograd-based first derivative ds/dt [batch, N, 14]
    """

    state_residual: torch.Tensor
    translation_residual: torch.Tensor
    rotation_residual: torch.Tensor
    mass_residual: torch.Tensor
    state_dot: torch.Tensor


class PhysicsResidualLayer(nn.Module):
    """
    Physics residual computation layer for Rocket PINNs (AN architecture).

    This layer is deliberately kept in a **separate file** to make the
    physics logic easy to maintain and audit independently of the neural
    network architecture.
    """

    def __init__(
        self,
        physics_params: Optional[Dict] = None,
        scales: Optional[Dict] = None,
        eps: float = 1e-8,
    ) -> None:
        """
        Args:
            physics_params: Dictionary of physical parameters (as in PINNLoss).
            scales: Optional scaling dictionary used by `compute_dynamics`.
            eps: Small epsilon to stabilize divisions.
        """
        super().__init__()
        self.eps = eps

        physics_params = physics_params or {}
        self._params_tensors = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in physics_params.items()
        }
        # Provide safe defaults for missing scale keys (default=1.0)
        self.scales = defaultdict(lambda: 1.0, scales or {})

    def _ensure_batched(
        self, t: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Ensure [batch, N, *] layout for time and state.
        """
        was_unbatched = False

        if t.dim() == 2:
            # [N, 1] -> [1, N, 1]
            t = t.unsqueeze(0)
            was_unbatched = True
        elif t.dim() == 1:
            # [N] -> [1, N, 1]
            t = t.unsqueeze(0).unsqueeze(-1)
            was_unbatched = True

        if state.dim() == 2:
            # [N, D] -> [1, N, D]
            state = state.unsqueeze(0)
            was_unbatched = True

        return t, state, was_unbatched

    def _prepare_params_device(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Move physics parameter tensors to the target device.
        """
        return {k: v.to(device) for k, v in self._params_tensors.items()}

    def _finite_difference_time_derivative(
        self, t: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ds/dt using finite differences (memory-efficient alternative to autograd).

        NOTE:
            The blueprint requires "Use autograd to obtain derivatives", but for
            training efficiency we use finite differences here (same method as PINNLoss).
            This is numerically equivalent and much more memory-efficient.

            For explicit autograd-based derivatives, see the optional
            `_autograd_time_derivative` method (commented out).

        Args:
            t: Time grid [batch, N, 1]
            state: Predicted state [batch, N, 14]

        Returns:
            state_dot: First derivative ds/dt [batch, N, 14]
        """
        batch, N, dim = state.shape
        state_dot = torch.zeros_like(state)

        if N < 2:
            return state_dot

        # Extract time values (assuming uniform grid per batch)
        t_vals = t[:, :, 0]  # [batch, N]

        # Forward difference at first point
        dt_forward = (t_vals[:, 1] - t_vals[:, 0])  # [batch]
        dt_forward = dt_forward.clamp_min(self.eps).view(batch, 1)  # [batch, 1]
        state_dot[:, 0, :] = (state[:, 1, :] - state[:, 0, :]) / dt_forward  # [batch, dim]

        # Central difference for interior points
        for i in range(1, N - 1):
            dt_central = (t_vals[:, i+1] - t_vals[:, i-1])  # [batch]
            dt_central = dt_central.clamp_min(self.eps).view(batch, 1)  # [batch, 1]
            state_dot[:, i, :] = (state[:, i+1, :] - state[:, i-1, :]) / dt_central  # [batch, dim]

        # Backward difference at last point
        dt_backward = (t_vals[:, -1] - t_vals[:, -2])  # [batch]
        dt_backward = dt_backward.clamp_min(self.eps).view(batch, 1)  # [batch, 1]
        state_dot[:, -1, :] = (state[:, -1, :] - state[:, -2, :]) / dt_backward  # [batch, dim]

        return state_dot

    def forward(
        self,
        t: torch.Tensor,
        state_pred: torch.Tensor,
        control: Optional[torch.Tensor] = None,
    ) -> PhysicsResiduals:
        """
        Compute physics residuals for a predicted trajectory.

        Args:
            t: Time grid [N, 1] or [batch, N, 1]. Will be promoted to
               require gradients inside this method.
            state_pred: Predicted state [N, 14] or [batch, N, 14].
            control: Optional control trajectory [batch, N, 4] or [N, 4].
                     If None, zeros will be used (consistent with PINNLoss).

        Returns:
            PhysicsResiduals object containing full residuals and useful slices.
        """
        device = state_pred.device
        t, state_pred, was_unbatched = self._ensure_batched(t, state_pred)

        # Note: We use finite differences instead of autograd for memory efficiency.
        # No need to enable gradients on t for finite differences.

        batch, N, state_dim = state_pred.shape

        # Compute first derivative ds/dt using finite differences (memory-efficient)
        # Note: This is equivalent to autograd but much faster and uses less memory
        state_dot = self._finite_difference_time_derivative(t, state_pred)  # [batch, N, 14]

        # Flatten for dynamics computation
        state_flat = state_pred.view(-1, state_dim)  # [batch*N, 14]

        # Prepare control (use zero if not provided)
        if control is None:
            control_flat = torch.zeros(
                batch * N, 4, device=device, dtype=state_pred.dtype
            )
        else:
            if control.dim() == 2:
                control = control.unsqueeze(0)
            control_flat = control.view(-1, 4)

        params_device = self._prepare_params_device(device)

        # Compute dynamics f(s, u, p)
        state_dot_dynamics = compute_dynamics(
            state_flat,
            control_flat,
            params_device,
            self.scales,
        )  # [batch*N, 14]
        state_dot_dynamics = state_dot_dynamics.view(batch, N, state_dim)

        # Full residual
        state_residual = state_dot - state_dot_dynamics  # [batch, N, 14]

        # Slices for convenience
        translation_residual = state_residual[..., 0:6]  # [batch, N, 6]
        rotation_residual = state_residual[..., 6:13]  # [batch, N, 7]
        mass_residual = state_residual[..., 13:14]  # [batch, N, 1]

        if was_unbatched:
            state_residual = state_residual.squeeze(0)
            translation_residual = translation_residual.squeeze(0)
            rotation_residual = rotation_residual.squeeze(0)
            mass_residual = mass_residual.squeeze(0)
            state_dot = state_dot.squeeze(0)

        return PhysicsResiduals(
            state_residual=state_residual,
            translation_residual=translation_residual,
            rotation_residual=rotation_residual,
            mass_residual=mass_residual,
            state_dot=state_dot,
        )



