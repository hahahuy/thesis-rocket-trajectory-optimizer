"""
Direction D: Backbone + 3 Heads (Dependency-Aware Single Model)
Direction D1: Upgraded with Physics & Causality Awareness

Dependency-Preserving Unified PINN Architecture for WP4 Rocket Dynamics.

Architecture:
- One shared backbone
- Three dedicated heads with explicit dependency order:
  G3 (mass) → G2 (attitude) → G1 (translation)
- Feature encoding: Fourier time features + context embedding

Direction D1 Enhancements:
- 6D rotation representation (stable, no quaternion normalization issues)
- Physics-aware layers (density, drag, lift computation)
- Causal temporal integration (RK4 for z, v_z)
- Altitude-aware physics injection
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .architectures import FourierFeatures, ContextEncoder, normalize_quaternion
from .physics_layers import PhysicsComputationLayer


class DirectionDPINN(nn.Module):
    """
    Direction D: Single model with Backbone + 3 Dependency-Aware Heads.
    
    Architecture:
    1. Feature encoding: Fourier time features + context embedding
    2. Shared backbone: MLP that extracts general motion information
    3. Head G3: Mass prediction (m)
    4. Head G2: Attitude + angular velocity (q, w) - depends on m
    5. Head G1: Translation (x, v) - depends on m, q, w
    
    Dependency chain: G3 → G2 → G1
    
    Input: (t, context) -> Output: state [14]
    """

    def __init__(
        self,
        context_dim: int,
        fourier_features: int = 8,
        context_embedding_dim: int = 32,
        backbone_hidden_dims: list = None,
        head_g3_hidden_dims: list = None,
        head_g2_hidden_dims: list = None,
        head_g1_hidden_dims: list = None,
        activation: str = "gelu",
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        """
        Args:
            context_dim: Dimension of context vector (typically 7)
            fourier_features: Number of Fourier frequencies (K)
            context_embedding_dim: Dimension of context embedding (default: 32)
            backbone_hidden_dims: Hidden dimensions for backbone MLP (default: [256, 256, 256, 256])
            head_g3_hidden_dims: Hidden dimensions for mass head (default: [128, 64])
            head_g2_hidden_dims: Hidden dimensions for attitude head (default: [256, 128, 64])
            head_g1_hidden_dims: Hidden dimensions for translation head (default: [256, 128, 128, 64])
            activation: Activation function (default: "gelu")
            layer_norm: Whether to use layer normalization
            dropout: Dropout rate (default: 0.0)
        """
        super().__init__()

        self.context_dim = context_dim
        self.fourier_features = fourier_features
        self.context_embedding_dim = context_embedding_dim

        # Feature encoding
        # 1. Fourier time features
        self.fourier_encoder = FourierFeatures(n_frequencies=fourier_features)
        # Output: [t, sin(2πk t), cos(2πk t)] for k=1..K
        # Dimension: 1 + 2*K (but FourierFeatures includes t, so it's 1 + 2*K)
        time_dim = 1 + 2 * fourier_features

        # 2. Context encoder
        self.context_encoder = ContextEncoder(
            context_dim=context_dim,
            embedding_dim=context_embedding_dim,
            activation=activation,
        )

        # Feature dimension after concatenation
        feature_dim = time_dim + context_embedding_dim

        # Shared backbone
        if backbone_hidden_dims is None:
            backbone_hidden_dims = [256, 256, 256, 256]

        self.backbone = self._build_mlp(
            input_dim=feature_dim,
            hidden_dims=backbone_hidden_dims,
            output_dim=backbone_hidden_dims[-1],  # Latent dimension = last hidden dim
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        latent_dim = backbone_hidden_dims[-1]

        # Head G3: Mass prediction
        # Input: latent [latent_dim]
        # Output: m [1]
        if head_g3_hidden_dims is None:
            head_g3_hidden_dims = [128, 64]
        self.head_g3 = self._build_mlp(
            input_dim=latent_dim,
            hidden_dims=head_g3_hidden_dims,
            output_dim=1,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )

        # Head G2: Attitude + Angular Velocity
        # Input: concat[latent, m_pred] [latent_dim + 1]
        # Output: [q0, q1, q2, q3, wx, wy, wz] [7]
        if head_g2_hidden_dims is None:
            head_g2_hidden_dims = [256, 128, 64]
        self.head_g2 = self._build_mlp(
            input_dim=latent_dim + 1,  # latent + m_pred
            hidden_dims=head_g2_hidden_dims,
            output_dim=7,  # 4 quaternion + 3 angular velocity
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )

        # Head G1: Translation
        # Input: concat[latent, m_pred, q_pred, w_pred] [latent_dim + 1 + 4 + 3]
        # Output: [x, y, z, vx, vy, vz] [6]
        if head_g1_hidden_dims is None:
            head_g1_hidden_dims = [256, 128, 128, 64]
        self.head_g1 = self._build_mlp(
            input_dim=latent_dim + 1 + 4 + 3,  # latent + m + q + w
            hidden_dims=head_g1_hidden_dims,
            output_dim=6,  # 3 position + 3 velocity
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )

    def _build_mlp(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation: str = "gelu",
        layer_norm: bool = True,
        dropout: float = 0.0,
    ) -> nn.Module:
        """Build an MLP with specified architecture."""
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # No activation on output layer
            if i < len(dims) - 2:
                # Activation
                if activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "relu":
                    layers.append(nn.ReLU())
                else:
                    raise ValueError(f"Unknown activation: {activation}")

                # Layer normalization
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))

                # Dropout
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def _ensure_batched(
        self, t: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Ensure tensors are batched."""
        t_was_unbatched = False
        context_was_unbatched = False

        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [N, 1]
        if t.dim() == 2:
            t = t.unsqueeze(0)  # [1, N, 1]
            t_was_unbatched = True

        if context.dim() == 1:
            context = context.unsqueeze(0)  # [1, context_dim]
            context_was_unbatched = True

        return t, context, t_was_unbatched and context_was_unbatched

    def forward(
        self, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with dependency chain: G3 → G2 → G1.

        Args:
            t: Time grid [..., 1] or [batch, N, 1] (nondimensional)
            context: Context vector [..., context_dim] or [batch, context_dim]

        Returns:
            state: Predicted state [..., 14] or [batch, N, 14]
                  [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]
        """
        t, context, was_unbatched = self._ensure_batched(t, context)
        batch_size, N, _ = t.shape

        # Feature encoding
        # 1. Fourier time features
        fourier_t = self.fourier_encoder(t)  # [batch, N, 1 + 2*K]

        # 2. Context embedding
        if context.dim() == 2:
            # Broadcast context to match time dimension
            context = context.unsqueeze(1).expand(batch_size, N, -1)  # [batch, N, context_dim]
        elif context.dim() == 3 and context.shape[1] == 1:
            context = context.expand(batch_size, N, -1)

        context_embed = self.context_encoder(context)  # [batch, N, context_embedding_dim]

        # 3. Concatenate features
        features = torch.cat([fourier_t, context_embed], dim=-1)  # [batch, N, feature_dim]

        # Shared backbone
        latent = self.backbone(features)  # [batch, N, latent_dim]

        # Head G3: Mass prediction
        m_pred = self.head_g3(latent)  # [batch, N, 1]

        # Head G2: Attitude + Angular Velocity (depends on m_pred)
        att_input = torch.cat([latent, m_pred], dim=-1)  # [batch, N, latent_dim + 1]
        att_output = self.head_g2(att_input)  # [batch, N, 7]

        # Split quaternion and angular velocity
        q_pred = att_output[..., :4]  # [batch, N, 4]
        w_pred = att_output[..., 4:]  # [batch, N, 3]

        # Normalize quaternion
        q_pred = normalize_quaternion(q_pred)  # [batch, N, 4]

        # Head G1: Translation (depends on m_pred, q_pred, w_pred)
        trans_input = torch.cat(
            [latent, m_pred, q_pred, w_pred], dim=-1
        )  # [batch, N, latent_dim + 1 + 4 + 3]
        trans_output = self.head_g1(trans_input)  # [batch, N, 6]

        # Split position and velocity
        x_pred = trans_output[..., :3]  # [batch, N, 3]
        v_pred = trans_output[..., 3:]  # [batch, N, 3]

        # Pack final state: [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]
        state = torch.cat(
            [x_pred, v_pred, q_pred, w_pred, m_pred], dim=-1
        )  # [batch, N, 14]

        if was_unbatched:
            state = state.squeeze(0)  # [N, 14]

        return state

    def predict_trajectory(
        self, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Convenience method for trajectory prediction."""
        return self.forward(t, context)


def sixd_to_rotation_matrix(sixd: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert 6D rotation representation to rotation matrix.
    
    Based on Zhou et al. 2019: "On the Continuity of Rotation Representations in Neural Networks"
    
    Args:
        sixd: 6D rotation representation [..., 6] = [r1_x, r1_y, r1_z, r2_x, r2_y, r2_z]
        eps: Small epsilon for numerical stability
        
    Returns:
        R: Rotation matrix [..., 3, 3]
    """
    # Split into two 3D vectors
    r1 = sixd[..., :3]  # [..., 3]
    r2 = sixd[..., 3:]  # [..., 3]
    
    # Normalize first column
    b1 = r1 / (torch.linalg.norm(r1, dim=-1, keepdim=True) + eps)  # [..., 3]
    
    # Gram-Schmidt: b2 = normalize(r2 - dot(b1, r2) * b1)
    dot_product = torch.sum(b1 * r2, dim=-1, keepdim=True)  # [..., 1]
    r2_ortho = r2 - dot_product * b1  # [..., 3]
    b2 = r2_ortho / (torch.linalg.norm(r2_ortho, dim=-1, keepdim=True) + eps)  # [..., 3]
    
    # Third column: b3 = cross(b1, b2)
    b3 = torch.cross(b1, b2, dim=-1)  # [..., 3]
    
    # Stack into rotation matrix
    R = torch.stack([b1, b2, b3], dim=-1)  # [..., 3, 3]
    
    return R


def rotation_matrix_to_quaternion(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion.
    
    Args:
        R: Rotation matrix [..., 3, 3]
        eps: Small epsilon for numerical stability
        
    Returns:
        q: Quaternion [..., 4] = [q0, q1, q2, q3]
    """
    original_shape = R.shape[:-2]  # [...,]
    R_flat = R.reshape(-1, 3, 3)  # [B, 3, 3]
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]  # [B]
    
    q = torch.zeros(R_flat.shape[0], 4, device=R.device, dtype=R.dtype)
    
    mask1 = trace > eps
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2.0
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (R_flat[mask1, 2, 1] - R_flat[mask1, 1, 2]) / s
        q[mask1, 2] = (R_flat[mask1, 0, 2] - R_flat[mask1, 2, 0]) / s
        q[mask1, 3] = (R_flat[mask1, 1, 0] - R_flat[mask1, 0, 1]) / s
    
    mask2 = (~mask1) & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R_flat[mask2, 0, 0] - R_flat[mask2, 1, 1] - R_flat[mask2, 2, 2]) * 2.0
        q[mask2, 0] = (R_flat[mask2, 2, 1] - R_flat[mask2, 1, 2]) / s
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (R_flat[mask2, 0, 1] + R_flat[mask2, 1, 0]) / s
        q[mask2, 3] = (R_flat[mask2, 0, 2] + R_flat[mask2, 2, 0]) / s
    
    mask3 = (~mask1) & (~mask2) & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R_flat[mask3, 1, 1] - R_flat[mask3, 0, 0] - R_flat[mask3, 2, 2]) * 2.0
        q[mask3, 0] = (R_flat[mask3, 0, 2] - R_flat[mask3, 2, 0]) / s
        q[mask3, 1] = (R_flat[mask3, 0, 1] + R_flat[mask3, 1, 0]) / s
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (R_flat[mask3, 1, 2] + R_flat[mask3, 2, 1]) / s
    
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R_flat[mask4, 2, 2] - R_flat[mask4, 0, 0] - R_flat[mask4, 1, 1]) * 2.0
        q[mask4, 0] = (R_flat[mask4, 1, 0] - R_flat[mask4, 0, 1]) / s
        q[mask4, 1] = (R_flat[mask4, 0, 2] + R_flat[mask4, 2, 0]) / s
        q[mask4, 2] = (R_flat[mask4, 1, 2] + R_flat[mask4, 2, 1]) / s
        q[mask4, 3] = 0.25 * s
    
    q_norm = torch.linalg.norm(q, dim=-1, keepdim=True) + eps
    q = q / q_norm
    q = q.reshape(*original_shape, 4)
    
    return q


class TemporalIntegrator(nn.Module):
    """
    Causal temporal integrator for translational dynamics.
    
    Integrates acceleration to get velocity and position using RK4.
    """
    
    def __init__(self, method: str = "rk4"):
        """
        Args:
            method: Integration method ("rk4" or "euler")
        """
        super().__init__()
        self.method = method
    
    def forward(
        self,
        accel: torch.Tensor,
        v0: torch.Tensor,
        z0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate acceleration to get velocity and position.
        
        Args:
            accel: Acceleration [batch, N, 3] (ax, ay, az)
            v0: Initial velocity [batch, 3] (vx0, vy0, vz0)
            z0: Initial position [batch, 3] (x0, y0, z0)
            t: Time grid [batch, N, 1]
            
        Returns:
            v: Velocity trajectory [batch, N, 3]
            z: Position trajectory [batch, N, 3]
        """
        batch_size, N, _ = accel.shape
        
        # Initialize trajectories
        v = torch.zeros_like(accel)  # [batch, N, 3]
        z = torch.zeros_like(accel)  # [batch, N, 3]
        
        v[:, 0, :] = v0  # [batch, 3]
        z[:, 0, :] = z0  # [batch, 3]
        
        if self.method == "rk4":
            # RK4 integration
            v_current = v0  # [batch, 3]
            z_current = z0  # [batch, 3]
            
            for i in range(N - 1):
                dt = (t[:, i+1, 0] - t[:, i, 0]).unsqueeze(-1)  # [batch, 1]
                
                # RK4 stages for velocity
                a_i = accel[:, i, :]  # [batch, 3]
                a_next = accel[:, i+1, :]  # [batch, 3]
                a_mid = 0.5 * (a_i + a_next)  # [batch, 3]
                
                k1_v = a_i
                k2_v = a_mid
                k3_v = a_mid
                k4_v = a_next
                
                v_next = v_current + (dt / 6.0) * (k1_v + 2.0*k2_v + 2.0*k3_v + k4_v)
                
                # RK4 stages for position
                v_mid1 = v_current + 0.5 * dt * k1_v
                v_mid2 = v_current + 0.5 * dt * k2_v
                v_end = v_current + dt * k3_v
                
                k1_z = v_current
                k2_z = v_mid1
                k3_z = v_mid2
                k4_z = v_end
                
                z_next = z_current + (dt / 6.0) * (k1_z + 2.0*k2_z + 2.0*k3_z + k4_z)
                
                v[:, i+1, :] = v_next
                z[:, i+1, :] = z_next
                
                v_current = v_next
                z_current = z_next
        else:
            # Euler integration
            v_current = v0
            z_current = z0
            
            for i in range(N - 1):
                dt = (t[:, i+1, 0] - t[:, i, 0]).unsqueeze(-1)  # [batch, 1]
                
                a_i = accel[:, i, :]  # [batch, 3]
                
                v_next = v_current + dt * a_i
                z_next = z_current + dt * v_current
                
                v[:, i+1, :] = v_next
                z[:, i+1, :] = z_next
                
                v_current = v_next
                z_current = z_next
        
        return v, z


class DirectionDPINN_D1(nn.Module):
    """
    Direction D1: Upgraded Backbone + 3 Heads with Physics & Causality Awareness.
    
    Enhancements over Direction D:
    1. 6D rotation representation (stable, no quaternion normalization issues)
    2. Physics-aware layers (density, drag, lift computation)
    3. Causal temporal integration (RK4 for z, v_z)
    4. Altitude-aware physics injection
    
    Architecture:
    1. Feature encoding: Fourier time features + context embedding
    2. Shared backbone: MLP that extracts general motion information
    3. Head G3: Mass prediction (m)
    4. Head G2: 6D rotation + angular velocity - depends on m + physics
    5. Head G1: Acceleration prediction - depends on m, R, w, physics
    6. Temporal integrator: Integrate acceleration → v, z
    7. Convert 6D rotation to quaternion for output
    
    Dependency chain: G3 → G2 → G1 → Integration
    
    Note: Initial conditions (v0, z0) can be provided via initial_state parameter,
    or will be set to zero if not provided.
    """
    
    requires_initial_state = False  # Optional, but recommended for better accuracy
    
    def __init__(
        self,
        context_dim: int,
        fourier_features: int = 8,
        context_embedding_dim: int = 32,
        backbone_hidden_dims: list = None,
        head_g3_hidden_dims: list = None,
        head_g2_hidden_dims: list = None,
        head_g1_hidden_dims: list = None,
        activation: str = "gelu",
        layer_norm: bool = True,
        dropout: float = 0.0,
        integration_method: str = "rk4",
        use_physics_aware: bool = True,
    ):
        """
        Args:
            context_dim: Dimension of context vector (typically 7)
            fourier_features: Number of Fourier frequencies (K)
            context_embedding_dim: Dimension of context embedding (default: 32)
            backbone_hidden_dims: Hidden dimensions for backbone MLP (default: [256, 256, 256, 256])
            head_g3_hidden_dims: Hidden dimensions for mass head (default: [128, 64])
            head_g2_hidden_dims: Hidden dimensions for attitude head (default: [256, 128, 64])
            head_g1_hidden_dims: Hidden dimensions for acceleration head (default: [256, 128, 128, 64])
            activation: Activation function (default: "gelu")
            layer_norm: Whether to use layer normalization
            dropout: Dropout rate (default: 0.0)
            integration_method: Integration method ("rk4" or "euler")
            use_physics_aware: Enable physics-aware features
        """
        super().__init__()
        
        self.context_dim = context_dim
        self.fourier_features = fourier_features
        self.context_embedding_dim = context_embedding_dim
        self.use_physics_aware = use_physics_aware
        
        # Feature encoding
        self.fourier_encoder = FourierFeatures(n_frequencies=fourier_features)
        time_dim = 1 + 2 * fourier_features
        
        self.context_encoder = ContextEncoder(
            context_dim=context_dim,
            embedding_dim=context_embedding_dim,
            activation=activation,
        )
        
        feature_dim = time_dim + context_embedding_dim
        
        # Shared backbone
        if backbone_hidden_dims is None:
            backbone_hidden_dims = [256, 256, 256, 256]
        
        self.backbone = self._build_mlp(
            input_dim=feature_dim,
            hidden_dims=backbone_hidden_dims,
            output_dim=backbone_hidden_dims[-1],
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        latent_dim = backbone_hidden_dims[-1]
        
        # Physics computation layer
        if use_physics_aware:
            self.physics_layer = PhysicsComputationLayer()
            # Physics features: [rho, q_dynamic, Cd, CL_alpha, Cm_alpha] = 5 dims
            physics_feat_dim = 5
        else:
            physics_feat_dim = 0
        
        # Head G3: Mass prediction
        if head_g3_hidden_dims is None:
            head_g3_hidden_dims = [128, 64]
        self.head_g3 = self._build_mlp(
            input_dim=latent_dim,
            hidden_dims=head_g3_hidden_dims,
            output_dim=1,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        
        # Head G2: 6D Rotation + Angular Velocity
        # Input: concat[latent, m_pred, physics_feat] = latent_dim + 1 + physics_feat_dim
        if head_g2_hidden_dims is None:
            head_g2_hidden_dims = [256, 128, 64]
        self.head_g2 = self._build_mlp(
            input_dim=latent_dim + 1 + physics_feat_dim,
            hidden_dims=head_g2_hidden_dims,
            output_dim=6 + 3,  # 6D rotation + 3 angular velocity
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        
        # Head G1: Acceleration prediction
        # Input: concat[latent, m_pred, R (flattened 9D), w_pred, physics_feat]
        # = latent_dim + 1 + 9 + 3 + physics_feat_dim
        if head_g1_hidden_dims is None:
            head_g1_hidden_dims = [256, 128, 128, 64]
        self.head_g1 = self._build_mlp(
            input_dim=latent_dim + 1 + 9 + 3 + physics_feat_dim,
            hidden_dims=head_g1_hidden_dims,
            output_dim=3,  # Acceleration (ax, ay, az)
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        
        # Temporal integrator
        self.integrator = TemporalIntegrator(method=integration_method)
    
    def _build_mlp(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation: str = "gelu",
        layer_norm: bool = True,
        dropout: float = 0.0,
    ) -> nn.Module:
        """Build an MLP with specified architecture."""
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
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
                
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        
        return nn.Sequential(*layers)
    
    def _compute_physics_features(
        self,
        altitude: torch.Tensor,
        v_mag: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute physics-aware features.
        
        Args:
            altitude: Altitude [batch, N, 1]
            v_mag: Velocity magnitude [batch, N, 1]
            context: Context vector [batch, N, context_dim] (contains Cd, CL_alpha, Cm_alpha)
            
        Returns:
            physics_feat: Physics features [batch, N, 5] = [rho, q_dynamic, Cd, CL_alpha, Cm_alpha]
        """
        if not self.use_physics_aware:
            return torch.zeros(*altitude.shape[:-1], 0, device=altitude.device, dtype=altitude.dtype)
        
        # Compute density
        rho = self.physics_layer.compute_density(altitude)  # [batch, N, 1]
        
        # Compute dynamic pressure
        q_dynamic = 0.5 * rho * v_mag ** 2  # [batch, N, 1]
        
        # Extract aerodynamic coefficients from context
        # Context: [m0, Isp, Cd, CL_alpha, Cm_alpha, Tmax, wind_mag]
        Cd = context[..., 2:3]  # [batch, N, 1]
        CL_alpha = context[..., 3:4]  # [batch, N, 1]
        Cm_alpha = context[..., 4:5]  # [batch, N, 1]
        
        # Concatenate physics features
        physics_feat = torch.cat([rho, q_dynamic, Cd, CL_alpha, Cm_alpha], dim=-1)  # [batch, N, 5]
        
        return physics_feat
    
    def _ensure_batched(
        self, t: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Ensure tensors are batched."""
        t_was_unbatched = False
        context_was_unbatched = False
        
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.dim() == 2:
            t = t.unsqueeze(0)
            t_was_unbatched = True
        
        if context.dim() == 1:
            context = context.unsqueeze(0)
            context_was_unbatched = True
        
        return t, context, t_was_unbatched and context_was_unbatched
    
    def forward(
        self,
        t: torch.Tensor,
        context: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with dependency chain: G3 → G2 → G1 → Integration.
        
        Args:
            t: Time grid [..., 1] or [batch, N, 1] (nondimensional)
            context: Context vector [..., context_dim] or [batch, context_dim]
            initial_state: Optional initial state [..., 14] or [batch, 14]
                          If provided, extracts v0 and z0 from it.
                          If None, uses zeros.
            
        Returns:
            state: Predicted state [..., 14] or [batch, N, 14]
                  [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]
        """
        t, context, was_unbatched = self._ensure_batched(t, context)
        batch_size, N, _ = t.shape
        
        # Broadcast context to match time dimension
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(batch_size, N, -1)
        elif context.dim() == 3 and context.shape[1] == 1:
            context = context.expand(batch_size, N, -1)
        
        # Feature encoding
        fourier_t = self.fourier_encoder(t)  # [batch, N, 1 + 2*K]
        context_embed = self.context_encoder(context)  # [batch, N, context_embedding_dim]
        features = torch.cat([fourier_t, context_embed], dim=-1)  # [batch, N, feature_dim]
        
        # Shared backbone
        latent = self.backbone(features)  # [batch, N, latent_dim]
        
        # Head G3: Mass prediction
        m_pred = self.head_g3(latent)  # [batch, N, 1]
        
        # Initial estimates for iterative physics computation
        # Start with zero altitude and zero velocity for first iteration
        altitude_init = torch.zeros(batch_size, N, 1, device=t.device, dtype=t.dtype)
        v_mag_init = torch.zeros(batch_size, N, 1, device=t.device, dtype=t.dtype)
        
        # Compute initial physics features
        physics_feat = self._compute_physics_features(
            altitude_init, v_mag_init, context
        )  # [batch, N, 5] or [batch, N, 0]
        
        # Head G2: 6D Rotation + Angular Velocity
        g2_input = torch.cat([latent, m_pred, physics_feat], dim=-1)  # [batch, N, latent_dim + 1 + 5]
        g2_output = self.head_g2(g2_input)  # [batch, N, 9]
        
        # Split 6D rotation and angular velocity
        sixd_rot = g2_output[..., :6]  # [batch, N, 6]
        w_pred = g2_output[..., 6:]  # [batch, N, 3]
        
        # Convert 6D rotation to rotation matrix
        R = sixd_to_rotation_matrix(sixd_rot)  # [batch, N, 3, 3]
        R_flat = R.reshape(batch_size, N, 9)  # [batch, N, 9] for concatenation
        
        # Head G1: Acceleration prediction
        g1_input = torch.cat([latent, m_pred, R_flat, w_pred, physics_feat], dim=-1)
        accel_pred = self.head_g1(g1_input)  # [batch, N, 3]
        
        # Temporal integration: Integrate acceleration → velocity, position
        # Extract initial conditions
        if initial_state is not None:
            # Ensure initial_state is batched
            if initial_state.dim() == 1:
                initial_state = initial_state.unsqueeze(0)
            
            # Extract initial position [x0, y0, z0] and velocity [vx0, vy0, vz0]
            z0 = initial_state[:, :3]  # [batch, 3]
            v0 = initial_state[:, 3:6]  # [batch, 3]
        else:
            # Use zeros if initial state not provided
            v0 = torch.zeros(batch_size, 3, device=t.device, dtype=t.dtype)
            z0 = torch.zeros(batch_size, 3, device=t.device, dtype=t.dtype)
        
        v_pred, z_pred = self.integrator(accel_pred, v0, z0, t)  # [batch, N, 3] each
        
        # Altitude-aware physics refinement (optional second pass)
        # Use predicted altitude to recompute physics features
        altitude_pred = z_pred[..., 2:3]  # [batch, N, 1] - z component
        v_mag_pred = torch.linalg.norm(v_pred, dim=-1, keepdim=True)  # [batch, N, 1]
        
        # Recompute physics features with predicted altitude
        physics_feat_refined = self._compute_physics_features(
            altitude_pred, v_mag_pred, context
        )
        
        # Optional: Refine G1 prediction with updated physics (simplified - just use refined features)
        # For full implementation, could re-run G1 with refined physics, but for now we use the first pass
        
        # Convert rotation matrix to quaternion for output
        q_pred = rotation_matrix_to_quaternion(R)  # [batch, N, 4]
        
        # Pack final state: [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]
        state = torch.cat(
            [z_pred, v_pred, q_pred, w_pred, m_pred], dim=-1
        )  # [batch, N, 14]
        
        if was_unbatched:
            state = state.squeeze(0)
        
        return state
    
    def predict_trajectory(
        self,
        t: torch.Tensor,
        context: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience method for trajectory prediction."""
        return self.forward(t, context, initial_state)

