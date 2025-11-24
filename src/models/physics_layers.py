"""
[PINN_V2][2025-XX-XX][C3 Architecture - Solution 1]
Physics computation layers for explicit physics-aware corrections.

Provides explicit computation of atmospheric density and drag forces
based on altitude and velocity, reducing the need for the model to
learn these relationships from data.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class PhysicsComputationLayer(nn.Module):
    """
    [PINN_V2][2025-XX-XX][C3 Architecture - Solution 1]
    Physics computation layer for explicit density and drag calculations.
    
    Computes:
    - Atmospheric density: rho(z) = rho0 * exp(-z/H)
    - Drag force: F_drag = 0.5 * rho * |v|² * Cd * S
    """

    def __init__(
        self,
        rho0: float = 1.225,
        h_scale: float = 8400.0,
        Cd: float = 0.3,
        S_ref: float = 0.05,
    ):
        """
        Args:
            rho0: Sea level density [kg/m³]
            h_scale: Atmospheric scale height [m]
            Cd: Drag coefficient
            S_ref: Reference area [m²]
        """
        super().__init__()
        self.register_buffer("rho0", torch.tensor(rho0, dtype=torch.float32))
        self.register_buffer("h_scale", torch.tensor(h_scale, dtype=torch.float32))
        self.register_buffer("Cd", torch.tensor(Cd, dtype=torch.float32))
        self.register_buffer("S_ref", torch.tensor(S_ref, dtype=torch.float32))

    def compute_density(self, altitude: torch.Tensor) -> torch.Tensor:
        """
        Compute atmospheric density from altitude.
        
        rho(z) = rho0 * exp(-z/H)
        
        Args:
            altitude: Altitude [..., 1] or [batch, N, 1] [m]
            
        Returns:
            density: Atmospheric density [..., 1] or [batch, N, 1] [kg/m³]
        """
        # Clamp altitude to non-negative for numerical stability
        altitude_clamped = torch.clamp(altitude, min=0.0)
        density = self.rho0 * torch.exp(-altitude_clamped / self.h_scale)
        return density

    def compute_drag_force(
        self, rho: torch.Tensor, v_mag: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute drag force magnitude.
        
        F_drag = 0.5 * rho * |v|² * Cd * S
        
        Args:
            rho: Atmospheric density [..., 1] or [batch, N, 1] [kg/m³]
            v_mag: Velocity magnitude [..., 1] or [batch, N, 1] [m/s]
            
        Returns:
            drag_force: Drag force magnitude [..., 1] or [batch, N, 1] [N]
        """
        # Dynamic pressure: q = 0.5 * rho * |v|²
        q = 0.5 * rho * v_mag ** 2
        # Drag force: F = q * Cd * S
        drag_force = q * self.Cd * self.S_ref
        return drag_force

    def compute_drag_acceleration(
        self, drag_force: torch.Tensor, mass: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute drag acceleration from drag force and mass.
        
        a_drag = F_drag / m
        
        Args:
            drag_force: Drag force [..., 1] or [batch, N, 1] [N]
            mass: Mass [..., 1] or [batch, N, 1] [kg]
            
        Returns:
            drag_acceleration: Drag acceleration [..., 1] or [batch, N, 1] [m/s²]
        """
        # Avoid division by zero
        mass_safe = torch.clamp(mass, min=1e-6)
        drag_acceleration = drag_force / mass_safe
        return drag_acceleration

