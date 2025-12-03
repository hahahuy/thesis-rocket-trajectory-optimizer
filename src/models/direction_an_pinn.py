"""
[PINN_V2][2025-12-01][AN Architecture]

Direction AN: Shared Stem + Mission Branches + Physics Residual Layer.

High-level layout (must be kept in this order):
1. Shared stem:
   - Input: normalized time and context (state/meta)
   - Modules: Fourier feature embedding + MLP with residual connections
   - Output: latent feature vector z
2. Mission branches:
   - Translation branch:   z -> [x, y, z, vx, vy, vz]
   - Rotation branch:      z -> [q0, q1, q2, q3, wx, wy, wz]
   - Mass / auxiliary:     z -> [m] (optional auxiliary branch)
3. Physics layer:
   - Not a neural network; computes physics residuals using autograd +
     existing rocket dynamics `compute_dynamics`.
   - Returns residual vectors that can be used in the loss.

Forward pass contract:
    input  -> shared stem      -> mission branches        -> physics layer
    (t, c) -> z                -> state_pred (14D state)  -> residuals

The model returns **both** predictions and residuals:
    state_pred, physics_residuals = model(t, context)

This class is implemented in addition to (and does not modify) existing
Direction D / D1 / D1.5 models.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .architectures import FourierFeatures, ContextEncoder, normalize_quaternion
from .branches import TranslationBranch, RotationBranch, MassBranch
from src.physics.physics_residual_layer import PhysicsResidualLayer, PhysicsResiduals


class ANSharedStem(nn.Module):
    """
    Shared stem for Direction AN.

    Design goals (per blueprint):
    - Input: normalized time and context vector
    - Fully connected layers with residual connections
    - Optional Fourier feature embedding on time
    - Layer normalization near the end
    - Moderate depth, stable width
    """

    def __init__(
        self,
        context_dim: int,
        fourier_features: int = 8,
        hidden_dim: int = 128,
        n_layers: int = 4,
        activation: str = "tanh",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.context_dim = context_dim
        self.fourier_features = fourier_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_layer_norm = use_layer_norm

        # Time encoding: Fourier features on normalized time
        self.time_embedding = FourierFeatures(n_frequencies=fourier_features)
        time_dim = 1 + 2 * fourier_features

        # Context encoder: simple MLP to a fixed embedding
        self.context_encoder = ContextEncoder(
            context_dim=context_dim,
            embedding_dim=hidden_dim,
            activation=activation,
        )

        input_dim = time_dim + hidden_dim

        # Build residual MLP stack
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_dim if i > 0 else input_dim, hidden_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
        self.mlp = nn.ModuleList(layers)

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None

    def _ensure_batched(
        self, t: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        t_was_unbatched = False

        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.dim() == 2:
            t = t.unsqueeze(0)
            t_was_unbatched = True

        if context.dim() == 1:
            context = context.unsqueeze(0)

        return t, context, t_was_unbatched

    def forward(self, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time [..., 1] or [batch, N, 1] (nondimensional)
            context: Context [..., context_dim] or [batch, context_dim]

        Returns:
            latent: Shared latent features [batch, N, hidden_dim] or [N, hidden_dim]
        """
        t, context, was_unbatched = self._ensure_batched(t, context)
        batch, N, _ = t.shape

        # Fourier time features
        t_emb = self.time_embedding(t)  # [batch, N, time_dim]

        # Broadcast and encode context
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(batch, N, -1)
        elif context.dim() == 3 and context.shape[1] == 1:
            context = context.expand(batch, N, -1)

        ctx_emb = self.context_encoder(context)  # [batch, N, hidden_dim]

        x = torch.cat([t_emb, ctx_emb], dim=-1)  # [batch, N, input_dim]

        # Residual MLP stack (position-wise)
        h = x
        for i in range(0, len(self.mlp), 2):
            lin = self.mlp[i]
            act = self.mlp[i + 1]
            h_new = act(lin(h))
            # Residual connection once dimensions match
            if h_new.shape == h.shape:
                h = h + h_new
            else:
                h = h_new

        if self.layer_norm is not None:
            h = self.layer_norm(h)

        if was_unbatched:
            h = h.squeeze(0)

        return h


class DirectionANPINN(nn.Module):
    """
    Direction AN: Shared stem + mission branches + physics residual layer.

    - Shared stem: `ANSharedStem`
    - Translation branch: position + velocity
    - Rotation branch: quaternion + angular rates
    - Mass branch: scalar mass
    - Physics layer: `PhysicsResidualLayer`

    Forward returns:
        state_pred, physics_residuals
    where:
        state_pred       : [..., 14] packed state
        physics_residuals: PhysicsResiduals dataclass
    """

    requires_initial_state = False

    def __init__(
        self,
        context_dim: int,
        fourier_features: int = 8,
        stem_hidden_dim: int = 128,
        stem_layers: int = 4,
        activation: str = "tanh",
        layer_norm: bool = True,
        translation_branch_dims: Optional[list] = None,
        rotation_branch_dims: Optional[list] = None,
        mass_branch_dims: Optional[list] = None,
        dropout: float = 0.0,
        physics_params: Optional[dict] = None,
        physics_scales: Optional[dict] = None,
    ) -> None:
        super().__init__()

        # 1. Shared stem
        self.stem = ANSharedStem(
            context_dim=context_dim,
            fourier_features=fourier_features,
            hidden_dim=stem_hidden_dim,
            n_layers=stem_layers,
            activation=activation,
            use_layer_norm=layer_norm,
        )

        # 2. Mission branches
        self.translation_branch = TranslationBranch(
            hidden_dim=stem_hidden_dim,
            branch_dims=translation_branch_dims or [128, 128],
            activation=activation,
            dropout=dropout,
        )
        self.rotation_branch = RotationBranch(
            hidden_dim=stem_hidden_dim,
            branch_dims=rotation_branch_dims or [256, 256],
            activation=activation,
            dropout=dropout,
        )
        self.mass_branch = MassBranch(
            hidden_dim=stem_hidden_dim,
            branch_dims=mass_branch_dims or [64],
            activation=activation,
            dropout=dropout,
        )

        # 3. Physics residual layer
        self.physics_layer = PhysicsResidualLayer(
            physics_params=physics_params,
            scales=physics_scales,
        )

    def forward(
        self,
        t: torch.Tensor,
        context: torch.Tensor,
        control: Optional[torch.Tensor] = None,
        T_mag: Optional[torch.Tensor] = None,
        q_dyn: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, PhysicsResiduals]:
        """
        Full forward pass.

        Args:
            t: Time [..., 1] or [batch, N, 1]
            context: Context [..., context_dim] or [batch, context_dim]
            control: Optional control trajectory [..., 4] or [batch, N, 4]

        Returns:
            state: Predicted state [..., 14]
            residuals: PhysicsResiduals object
        """
        # Use the same time tensor for physics layer (finite differences don't need gradients)
        t_model = t

        # 1. Shared stem
        latent = self.stem(t_model, context)  # [batch, N, hidden_dim] or [N, hidden_dim]

        # Ensure batched layout for branches and physics
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
            t_model = t_model.unsqueeze(0) if t_model.dim() == 2 else t_model
            context = context.unsqueeze(0) if context.dim() == 2 else context

        # 2. Mission branches (independent subnetworks)
        trans_out = self.translation_branch(latent)  # [batch, N, 6]
        rot_out = self.rotation_branch(latent)  # [batch, N, 7]
        mass_out = self.mass_branch(latent)  # [batch, N, 1]

        # Split rotation into quaternion + angular rates and normalize quaternion
        q_pred = rot_out[..., :4]  # [batch, N, 4]
        w_pred = rot_out[..., 4:]  # [batch, N, 3]
        q_pred = normalize_quaternion(q_pred)

        # Translation split
        x_pred = trans_out[..., :3]  # [batch, N, 3]
        v_pred = trans_out[..., 3:]  # [batch, N, 3]

        # Pack final state
        state = torch.cat(
            [x_pred, v_pred, q_pred, w_pred, mass_out],
            dim=-1,
        )  # [batch, N, 14]

        # 3. Physics residuals (autograd-based)
        residuals = self.physics_layer(t_model, state, control=control)

        return state, residuals

    def predict_trajectory(
        self,
        t: torch.Tensor,
        context: torch.Tensor,
        control: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convenience wrapper returning only the state prediction.
        """
        state, _ = self.forward(t, context, control)
        return state



