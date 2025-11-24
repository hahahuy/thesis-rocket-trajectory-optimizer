"""
[PINN_V2][2025-XX-XX][C3 Architecture - Solution 5]
Cross-branch coordination module for aerodynamic coupling.

Coordinates translation, rotation, and mass branches through
aerodynamic coupling corrections.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .physics_layers import PhysicsComputationLayer


class AerodynamicCouplingModule(nn.Module):
    """
    [PINN_V2][2025-XX-XX][C3 Architecture - Solution 5]
    Aerodynamic coupling module that computes drag corrections
    based on velocity, orientation, and density.
    
    Input: [|v|, q0, q1, q2, q3, rho] → Output: drag_correction [6D]
    """

    def __init__(
        self,
        input_dim: int = 6,  # |v| (1) + quaternion (4) + rho (1)
        hidden_dims: list = None,
        output_dim: int = 6,  # [dx, dy, dz, dvx, dvy, dvz]
        activation: str = "gelu",
        dropout: float = 0.05,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        dims = [input_dim] + hidden_dims + [output_dim]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "relu":
                    layers.append(nn.ReLU())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        v_mag: torch.Tensor,
        quaternion: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute drag correction to translation.

        Args:
            v_mag: Velocity magnitude [..., 1] or [batch, N, 1]
            quaternion: Quaternion [..., 4] or [batch, N, 4] [q0, q1, q2, q3]
            rho: Atmospheric density [..., 1] or [batch, N, 1]

        Returns:
            drag_correction: Translation correction [..., 6] or [batch, N, 6]
                            [dx, dy, dz, dvx, dvy, dvz]
        """
        # Concatenate coupling variables
        coupling_input = torch.cat([v_mag, quaternion, rho], dim=-1)  # [..., 6]

        # Compute drag correction
        drag_correction = self.net(coupling_input)  # [..., 6]

        return drag_correction


class CoordinatedBranches(nn.Module):
    """
    [PINN_V2][2025-XX-XX][C3 Architecture - Solution 5]
    Coordinates translation, rotation, and mass branches through
    aerodynamic coupling.

    Architecture:
    1. Initial predictions from all branches
    2. Extract coupling variables: |v|, q, rho
    3. Compute aerodynamic coupling correction
    4. Apply correction to translation
    5. Return coordinated outputs
    """

    def __init__(
        self,
        translation_branch: nn.Module,
        rotation_branch: nn.Module,
        mass_branch: nn.Module,
        coupling_module: Optional[nn.Module] = None,
        physics_layer: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.translation_branch = translation_branch
        self.rotation_branch = rotation_branch
        self.mass_branch = mass_branch

        # Aerodynamic coupling module
        if coupling_module is None:
            self.coupling_module = AerodynamicCouplingModule()
        else:
            self.coupling_module = coupling_module

        # Physics computation layer
        if physics_layer is None:
            self.physics_layer = PhysicsComputationLayer()
        else:
            self.physics_layer = physics_layer

    def forward(
        self,
        z_traj: torch.Tensor,
        m0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with branch coordination.

        Args:
            z_traj: Latent trajectory [batch, N, latent_dim]
            m0: Initial mass [batch, 1]

        Returns:
            translation: [batch, N, 6] (coordinated)
            rotation: [batch, N, 7] [q0, q1, q2, q3, wx, wy, wz]
            mass: [batch, N, 1] (monotonically decreasing)
        """
        # 1. Initial predictions from all branches
        # Predict mass first (needed for translation correction)
        mass = self.mass_branch(z_traj, m0)  # [batch, N, 1]

        # Predict translation (with physics-aware correction if applicable)
        if hasattr(self.translation_branch, "forward") and "mass" in str(
            self.translation_branch.forward.__code__.co_varnames
        ):
            # PhysicsAwareTranslationBranch accepts mass
            translation_init = self.translation_branch(z_traj, mass)  # [batch, N, 6]
        else:
            # Standard TranslationBranch
            translation_init = self.translation_branch(z_traj)  # [batch, N, 6]

        # Predict rotation
        rotation = self.rotation_branch(z_traj)  # [batch, N, 7]

        # 2. Extract coupling variables
        # Extract velocity magnitude
        velocity = translation_init[..., 3:6]  # [batch, N, 3]
        v_mag = torch.linalg.norm(velocity, dim=-1, keepdim=True)  # [batch, N, 1]

        # Extract quaternion
        quaternion = rotation[..., :4]  # [batch, N, 4]

        # Extract altitude for density computation
        altitude = translation_init[..., 2:3]  # [batch, N, 1]
        rho = self.physics_layer.compute_density(altitude)  # [batch, N, 1]

        # 3. Compute aerodynamic coupling correction
        drag_correction = self.coupling_module(
            v_mag, quaternion, rho
        )  # [batch, N, 6]

        # 4. Apply correction to translation
        translation = translation_init + drag_correction  # [batch, N, 6]

        # 5. Return coordinated outputs
        return translation, rotation, mass

