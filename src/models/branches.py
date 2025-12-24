"""
[PINN_V2][2025-11-19][C2 Architecture]
[PINN_V2][2025-XX-XX][C3 Architecture]
Dedicated branches for translation, rotation, and mass predictions.

Each branch is a specialized MLP that processes the shared embedding
to produce subsystem-specific outputs.
"""

import torch
import torch.nn as nn
from typing import Optional


class TranslationBranch(nn.Module):
    """
    [PINN_V2][2025-11-19][C2 Architecture]
    Dedicated branch for translation (position + velocity).

    Output: [x, y, z, vx, vy, vz] (6D)

    NOTE:
        This branch is used by architectures that predict full translation
        state directly. Direction AN/AN1/AN2 use a reduced translation head
        (`TranslationBranchReducedXYFree`) that removes x,y from the neural
        output space and reconstructs them via velocity integration.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        branch_dims: list = None,
        activation: str = "gelu",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        if branch_dims is None:
            branch_dims = [128, 128]

        dims = [hidden_dim] + branch_dims + [6]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on output
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Shared embedding [..., hidden_dim] or [batch, N, hidden_dim]

        Returns:
            translation: [..., 6] or [batch, N, 6]
        """
        return self.net(z)


class TranslationBranchReducedXYFree(nn.Module):
    """
    [PINN_V2][2025-12-24][AN Architecture Update]
    Translation branch that predicts only horizontal velocities and vertical
    position/velocity.

    Output: [vx, vy, z, vz] (4D)

    Design intent:
        - Remove x, y from the neural output space
        - Enforce x(t), y(t) via deterministic integration of vx, vy
        - Reduce redundant degrees of freedom and stabilize position
          reconstruction
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        branch_dims: list = None,
        activation: str = "gelu",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        if branch_dims is None:
            branch_dims = [128, 128]

        dims = [hidden_dim] + branch_dims + [4]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on output
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Shared embedding [..., hidden_dim] or [batch, N, hidden_dim]

        Returns:
            translation_reduced: [..., 4] or [batch, N, 4]
                [vx, vy, z, vz]
        """
        return self.net(z)


class RotationBranch(nn.Module):
    """
    [PINN_V2][2025-11-19][C2 Architecture]
    Dedicated branch for rotation (quaternion + angular velocity).

    Output: [q0, q1, q2, q3, wx, wy, wz] (7D)
    Note: Quaternion normalization is applied externally (Set#1).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        branch_dims: list = None,
        activation: str = "gelu",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        if branch_dims is None:
            branch_dims = [256, 256]  # Wider for rotation complexity

        dims = [hidden_dim] + branch_dims + [7]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on output
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Shared embedding [..., hidden_dim] or [batch, N, hidden_dim]

        Returns:
            rotation: [..., 7] or [batch, N, 7]
            Note: Quaternion (first 4 dims) should be normalized externally.
        """
        return self.net(z)


class MassBranch(nn.Module):
    """
    [PINN_V2][2025-11-19][C2 Architecture]
    Dedicated branch for mass prediction.

    Output: [Δm] (1D, mass delta for Set#1 Δ-state reconstruction)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        branch_dims: list = None,
        activation: str = "gelu",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        if branch_dims is None:
            branch_dims = [64]  # Narrower for mass simplicity

        dims = [hidden_dim] + branch_dims + [1]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on output
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Shared embedding [..., hidden_dim] or [batch, N, hidden_dim]

        Returns:
            mass_delta: [..., 1] or [batch, N, 1]
        """
        return self.net(z)


def rotation_vector_to_quaternion(rotation_vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    [PINN_V2][2025-XX-XX][C3 Architecture - Solution 2]
    Convert rotation vector (axis-angle representation) to quaternion.
    Always produces unit quaternion by construction.
    
    Args:
        rotation_vec: Rotation vector [..., 3] (axis * angle)
        eps: Small epsilon for numerical stability
        
    Returns:
        quaternion: Unit quaternion [..., 4] [q0, q1, q2, q3]
    """
    # Compute rotation angle: ||rotation_vec||
    angle = torch.linalg.norm(rotation_vec, dim=-1, keepdim=True)  # [..., 1]
    
    # Compute axis: rotation_vec / angle
    axis = rotation_vec / (angle + eps)  # [..., 3]
    
    # Half angle for quaternion conversion
    half_angle = angle / 2.0  # [..., 1]
    
    # Quaternion components
    q0 = torch.cos(half_angle)  # [..., 1]
    q_vec = torch.sin(half_angle) * axis  # [..., 3]
    
    # Concatenate: [q0, q1, q2, q3]
    quaternion = torch.cat([q0, q_vec], dim=-1)  # [..., 4]
    
    # Normalize for numerical stability (should already be unit, but ensure)
    quat_norm = torch.linalg.norm(quaternion, dim=-1, keepdim=True) + eps
    quaternion = quaternion / quat_norm
    
    return quaternion


class RotationBranchMinimal(nn.Module):
    """
    [PINN_V2][2025-XX-XX][C3 Architecture - Solution 2]
    Dedicated branch for rotation using minimal representation.
    
    Outputs rotation vector [3D] → converts to quaternion [4D] (always unit norm).
    Angular velocity: [wx, wy, wz] (3D)
    Total output: [q0, q1, q2, q3, wx, wy, wz] (7D)
    
    Key advantage: Quaternion always unit norm by construction, no external normalization needed.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        branch_dims: list = None,
        activation: str = "gelu",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        if branch_dims is None:
            branch_dims = [256, 256]  # Wider for rotation complexity

        # Rotation vector MLP: outputs [rx, ry, rz] (3D)
        dims_rot = [hidden_dim] + branch_dims + [3]
        layers_rot = []
        for i in range(len(dims_rot) - 1):
            layers_rot.append(nn.Linear(dims_rot[i], dims_rot[i + 1]))
            if i < len(dims_rot) - 2:
                if activation == "gelu":
                    layers_rot.append(nn.GELU())
                elif activation == "tanh":
                    layers_rot.append(nn.Tanh())
                elif activation == "relu":
                    layers_rot.append(nn.ReLU())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
                if dropout > 0.0:
                    layers_rot.append(nn.Dropout(dropout))
        self.rotation_vec_mlp = nn.Sequential(*layers_rot)

        # Angular velocity MLP: outputs [wx, wy, wz] (3D)
        dims_ang = [hidden_dim] + branch_dims + [3]
        layers_ang = []
        for i in range(len(dims_ang) - 1):
            layers_ang.append(nn.Linear(dims_ang[i], dims_ang[i + 1]))
            if i < len(dims_ang) - 2:
                if activation == "gelu":
                    layers_ang.append(nn.GELU())
                elif activation == "tanh":
                    layers_ang.append(nn.Tanh())
                elif activation == "relu":
                    layers_ang.append(nn.ReLU())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
                if dropout > 0.0:
                    layers_ang.append(nn.Dropout(dropout))
        self.angular_vel_mlp = nn.Sequential(*layers_ang)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Shared embedding [..., hidden_dim] or [batch, N, hidden_dim]

        Returns:
            rotation: [..., 7] or [batch, N, 7] [q0, q1, q2, q3, wx, wy, wz]
            Quaternion is always unit norm by construction.
        """
        # Predict rotation vector
        rotation_vec = self.rotation_vec_mlp(z)  # [..., 3]
        
        # Convert to quaternion (always unit norm)
        quaternion = rotation_vector_to_quaternion(rotation_vec)  # [..., 4]
        
        # Predict angular velocity
        angular_vel = self.angular_vel_mlp(z)  # [..., 3]
        
        # Concatenate: [q0, q1, q2, q3, wx, wy, wz]
        rotation = torch.cat([quaternion, angular_vel], dim=-1)  # [..., 7]
        
        return rotation


class MonotonicMassBranch(nn.Module):
    """
    [PINN_V2][2025-XX-XX][C3 Architecture - Solution 3]
    Dedicated branch for mass prediction with structural monotonicity guarantee.
    
    Output: [m] (1D, absolute mass, always decreasing)
    
    Architecture:
    1. MLP predicts raw mass delta
    2. Apply -ReLU to ensure mass_delta <= 0 (always decreasing)
    3. Cumulative sum + initial mass → absolute mass
    
    Key advantage: Mass always decreases structurally, no violations possible.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        branch_dims: list = None,
        activation: str = "gelu",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        if branch_dims is None:
            branch_dims = [64]  # Narrower for mass simplicity

        dims = [hidden_dim] + branch_dims + [1]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on output
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

        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, m0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Shared embedding [..., hidden_dim] or [batch, N, hidden_dim]
            m0: Initial mass [..., 1] or [batch, 1]

        Returns:
            mass: Absolute mass [..., 1] or [batch, N, 1] (always decreasing)
        """
        # Predict raw mass delta (can be positive or negative)
        mass_delta_raw = self.mlp(z)  # [..., 1] or [batch, N, 1]
        
        # Apply -ReLU to ensure mass_delta <= 0 (always decreasing)
        mass_delta = -torch.relu(mass_delta_raw)  # [..., 1] or [batch, N, 1]
        
        # Ensure m0 has correct shape for broadcasting
        if z.dim() == 3:  # [batch, N, hidden_dim]
            batch_size, N, _ = z.shape
            if m0.dim() == 2:  # [batch, 1]
                m0 = m0.unsqueeze(1)  # [batch, 1, 1]
            elif m0.dim() == 1:  # [batch]
                m0 = m0.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
            # Cumulative sum along time dimension (dim=1)
            mass = torch.cumsum(mass_delta, dim=1) + m0  # [batch, N, 1]
        else:
            # For unbatched case, cumsum along last dimension
            mass = torch.cumsum(mass_delta, dim=-1) + m0  # [N, 1] or [1]
        
        return mass


class PhysicsAwareTranslationBranch(nn.Module):
    """
    [PINN_V2][2025-XX-XX][C3 Architecture - Solution 1]
    Physics-aware translation branch with explicit density and drag computation.
    
    Architecture:
    1. Standard MLP predicts initial translation [x, y, z, vx, vy, vz]
    2. Extract altitude z and velocity magnitude |v|
    3. Compute density: rho(z) = rho0 * exp(-z/H)
    4. Compute drag force: F_drag = 0.5 * rho * |v|² * Cd * S
    5. Apply physics-aware correction to vertical velocity: vz_corrected = vz - drag_z / m
    
    Output: [x, y, z, vx, vy, vz_corrected] (6D)
    
    Key advantage: Explicit physics computation reduces need to learn altitude→density→drag chain.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        branch_dims: list = None,
        activation: str = "gelu",
        dropout: float = 0.05,
        physics_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if branch_dims is None:
            branch_dims = [128, 128]

        # Standard MLP for initial translation prediction
        dims = [hidden_dim] + branch_dims + [6]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on output
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

        self.mlp = nn.Sequential(*layers)

        # Physics computation layer
        if physics_layer is None:
            from .physics_layers import PhysicsComputationLayer
            self.physics_layer = PhysicsComputationLayer()
        else:
            self.physics_layer = physics_layer

    def forward(
        self,
        z: torch.Tensor,
        mass: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z: Shared embedding [..., hidden_dim] or [batch, N, hidden_dim]
            mass: Mass [..., 1] or [batch, N, 1] (optional, for drag correction)

        Returns:
            translation: [..., 6] or [batch, N, 6] [x, y, z, vx, vy, vz_corrected]
        """
        # Initial translation prediction
        translation_raw = self.mlp(z)  # [..., 6] or [batch, N, 6]

        # Extract altitude and velocity components
        # translation_raw: [x, y, z, vx, vy, vz]
        position = translation_raw[..., :3]  # [..., 3] or [batch, N, 3]
        velocity = translation_raw[..., 3:6]  # [..., 3] or [batch, N, 3]

        # Extract altitude (z-component)
        altitude = position[..., 2:3]  # [..., 1] or [batch, N, 1]

        # Compute velocity magnitude
        v_mag = torch.linalg.norm(velocity, dim=-1, keepdim=True)  # [..., 1] or [batch, N, 1]

        # Compute density from altitude
        rho = self.physics_layer.compute_density(altitude)  # [..., 1] or [batch, N, 1]

        # Compute drag force
        drag_force = self.physics_layer.compute_drag_force(rho, v_mag)  # [..., 1] or [batch, N, 1]

        # Apply physics-aware correction to vertical velocity
        # If mass is provided, use it; otherwise, skip drag correction
        if mass is not None:
            # Compute drag acceleration
            drag_acceleration = self.physics_layer.compute_drag_acceleration(
                drag_force, mass
            )  # [..., 1] or [batch, N, 1]

            # Extract vertical velocity
            vz = velocity[..., 2:3]  # [..., 1] or [batch, N, 1]

            # Apply correction: vz_corrected = vz - drag_acceleration
            # (drag opposes motion, so subtract)
            vz_corrected = vz - drag_acceleration

            # Reconstruct velocity with corrected vz
            velocity_corrected = torch.cat(
                [velocity[..., :2], vz_corrected], dim=-1
            )  # [..., 3] or [batch, N, 3]

            # Reconstruct translation
            translation = torch.cat(
                [position, velocity_corrected], dim=-1
            )  # [..., 6] or [batch, N, 6]
        else:
            # No mass provided, return raw prediction
            translation = translation_raw

        return translation


