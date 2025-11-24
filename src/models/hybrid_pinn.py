"""
[PINN_V2][2025-11-18][Direction C]
Hybrid PINN combining a Transformer-based encoder for the latent
initial state with latent ODE dynamics.
"""

from typing import Tuple

import torch
import torch.nn as nn

from .architectures import (
    ContextEncoder,
    DeepContextEncoder,
    MLP,
    OutputHeads,
    TimeEmbedding,
    normalize_quaternion,
)
from .branches import (
    MassBranch,
    RotationBranch,
    TranslationBranch,
    MonotonicMassBranch,
    RotationBranchMinimal,
    PhysicsAwareTranslationBranch,
)
from .latent_ode import LatentDynamicsNet, LatentODEBlock, LatentODEBlockRK4
from .coordination import CoordinatedBranches
from .physics_layers import PhysicsComputationLayer
from .shared_stem import SharedStem


class RocketHybridPINN(nn.Module):
    """
    [PINN_V2][2025-11-18][Direction C]
    Hybrid architecture:
      1. Transformer encoder processes early time steps + context to infer z0
      2. Latent ODE evolves z(t)
      3. MLP decoder maps latent trajectory to physical state
    """

    def __init__(
        self,
        context_dim: int,
        latent_dim: int = 64,
        context_embedding_dim: int = 64,
        fourier_features: int = 8,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        encoder_window: int = 10,
        activation: str = "tanh",
        transformer_activation: str = "gelu",
        dynamics_n_hidden: int = 3,
        dynamics_n_neurons: int = 128,
        decoder_n_hidden: int = 3,
        decoder_n_neurons: int = 128,
        layer_norm: bool = True,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        self.encoder_window = encoder_window
        self.latent_dim = latent_dim
        self.context_dim = context_dim

        # Embeddings
        self.time_embedding = TimeEmbedding(n_frequencies=fourier_features)
        time_dim = 1 + 2 * fourier_features

        self.context_encoder = ContextEncoder(
            context_dim=context_dim,
            embedding_dim=context_embedding_dim,
            activation=activation,
        )

        # Sequence encoder to derive z0
        self.sequence_input_proj = nn.Linear(
            time_dim + context_embedding_dim, d_model
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=transformer_activation,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )
        self.z0_proj = nn.Linear(d_model, latent_dim)

        # Latent dynamics
        condition_dim = time_dim + context_embedding_dim
        self.dynamics_net = LatentDynamicsNet(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            n_hidden=dynamics_n_hidden,
            n_neurons=dynamics_n_neurons,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        self.ode_block = LatentODEBlock(self.dynamics_net)

        # Decoder
        hidden_dims = [decoder_n_neurons] * decoder_n_hidden
        self.decoder = MLP(
            input_dim=latent_dim,
            output_dim=14,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )

    def _ensure_batched(
        self, t: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
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
        self, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        t, context, was_unbatched = self._ensure_batched(t, context)
        batch_size, N, _ = t.shape

        # Broadcast context
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(batch_size, N, -1)

        # Embeddings
        t_emb = self.time_embedding(t)
        ctx_emb = self.context_encoder(context)
        seq_features = torch.cat([t_emb, ctx_emb], dim=-1)

        # Transformer encoder on early window
        window = min(self.encoder_window, N)
        enc_tokens = self.sequence_input_proj(seq_features[:, :window, :])
        enc_tokens = self.sequence_encoder(enc_tokens)
        z0_tokens = enc_tokens.mean(dim=1)
        z0 = self.z0_proj(z0_tokens)

        # Latent ODE integration on full grid
        condition = seq_features  # already [batch, N, time+ctx]
        z_traj = self.ode_block(z0, t, condition)
        state = self.decoder(z_traj)

        if was_unbatched:
            state = state.squeeze(0)

        return state

    def predict_trajectory(
        self, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(t, context)


class RocketHybridPINNC1(nn.Module):
    """
    [PINN_V2][2025-11-19][Direction C1]
    Hybrid architecture with output stability + deep context encoder enhancements.
    """

    requires_initial_state = True

    def __init__(
        self,
        context_dim: int,
        latent_dim: int = 64,
        context_embedding_dim: int = 32,
        fourier_features: int = 8,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        encoder_window: int = 10,
        activation: str = "tanh",
        transformer_activation: str = "gelu",
        dynamics_n_hidden: int = 3,
        dynamics_n_neurons: int = 128,
        decoder_n_hidden: int = 3,
        decoder_n_neurons: int = 128,
        layer_norm: bool = True,
        dropout: float = 0.05,
        debug_stats: bool = True,
    ) -> None:
        super().__init__()

        self.encoder_window = encoder_window
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.context_embedding_dim = context_embedding_dim
        self.debug_stats_enabled = debug_stats
        self._debug_stats = {}

        # Embeddings
        self.time_embedding = TimeEmbedding(n_frequencies=fourier_features)
        time_dim = 1 + 2 * fourier_features

        self.context_encoder = DeepContextEncoder(
            context_dim=context_dim,
            hidden_dims=(64, 128, 128, 64),
            output_dim=context_embedding_dim,
        )

        # Sequence encoder to derive z0
        self.sequence_input_proj = nn.Linear(
            time_dim + context_embedding_dim, d_model
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=transformer_activation,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )
        self.z0_proj = nn.Linear(d_model, latent_dim)

        # Latent dynamics
        condition_dim = time_dim + context_embedding_dim
        self.dynamics_net = LatentDynamicsNet(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            n_hidden=dynamics_n_hidden,
            n_neurons=dynamics_n_neurons,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        self.ode_block = LatentODEBlock(self.dynamics_net)

        # Decoder + split heads
        hidden_dims = [decoder_n_neurons] * decoder_n_hidden
        self.decoder = MLP(
            input_dim=latent_dim,
            output_dim=decoder_n_neurons,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        self.output_heads = OutputHeads(decoder_n_neurons)

    def _ensure_batched(
        self,
        t: torch.Tensor,
        context: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        was_unbatched = False

        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.dim() == 2:
            t = t.unsqueeze(0)
            was_unbatched = True

        if context.dim() == 1:
            context = context.unsqueeze(0)

        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)

        return t, context, initial_state, was_unbatched

    def get_debug_stats(self, reset: bool = True) -> dict:
        stats = getattr(self, "_debug_stats", {})
        if reset:
            self._debug_stats = {}
        return stats

    def _capture_debug_stats(
        self,
        ctx_emb: torch.Tensor,
        quat_raw: torch.Tensor,
        quat_norm: torch.Tensor,
        mass: torch.Tensor,
        delta_state: torch.Tensor,
    ) -> None:
        if not self.debug_stats_enabled:
            return

        with torch.no_grad():
            delta_norm = torch.linalg.norm(delta_state, dim=-1)
            quat_raw_norm = torch.linalg.norm(quat_raw, dim=-1)
            quat_norm_val = torch.linalg.norm(quat_norm, dim=-1)
            if mass.shape[1] > 1:
                mass_diff = torch.diff(mass, dim=1)
                mass_violation = (mass_diff > 1e-5).float().mean().item()
            else:
                mass_violation = 0.0

            ctx_stats = {
                "mean": ctx_emb.mean().item(),
                "std": ctx_emb.std(unbiased=False).item(),
                "min": ctx_emb.min().item(),
                "max": ctx_emb.max().item(),
            }

            self._debug_stats = {
                "quat_norm_raw_mean": float(quat_raw_norm.mean().item()),
                "quat_norm_raw_std": float(quat_raw_norm.std(unbiased=False).item()),
                "quat_norm_mean": float(quat_norm_val.mean().item()),
                "mass_increase_ratio": float(mass_violation),
                "delta_state_l2_mean": float(delta_norm.mean().item()),
                "delta_state_l2_max": float(delta_norm.max().item()),
                "context_embedding_stats": ctx_stats,
            }

    def forward(
        self,
        t: torch.Tensor,
        context: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        if initial_state is None:
            raise ValueError("RocketHybridPINNC1 requires the true initial state s0.")

        t, context, initial_state, was_unbatched = self._ensure_batched(
            t, context, initial_state
        )
        batch_size, N, _ = t.shape

        # Broadcast context across time if needed
        if context.dim() == 2:
            context = context[:, None, :].expand(batch_size, N, -1)
        elif context.dim() == 3 and context.shape[1] == 1:
            context = context.expand(batch_size, N, -1)

        # Embeddings
        t_emb = self.time_embedding(t)
        ctx_emb = self.context_encoder(context)
        seq_features = torch.cat([t_emb, ctx_emb], dim=-1)

        # Transformer encoder on early window
        window = min(self.encoder_window, N)
        enc_tokens = self.sequence_input_proj(seq_features[:, :window, :])
        enc_tokens = self.sequence_encoder(enc_tokens)
        z0_tokens = enc_tokens.mean(dim=1)
        z0 = self.z0_proj(z0_tokens)

        # Latent ODE integration on full grid
        condition = seq_features
        z_traj = self.ode_block(z0, t, condition)

        # Decoder features + split heads
        decoder_features = self.decoder(z_traj)
        translation, rotation_raw, mass_delta = self.output_heads(decoder_features)

        quat_raw = rotation_raw[..., :4]
        ang_vel = rotation_raw[..., 4:]
        quat_norm = normalize_quaternion(quat_raw)
        rotation = torch.cat([quat_norm, ang_vel], dim=-1)

        state_delta = torch.cat([translation, rotation, mass_delta], dim=-1)

        s0 = initial_state.unsqueeze(1).expand(-1, N, -1)
        state = s0 + state_delta

        self._capture_debug_stats(ctx_emb, quat_raw, quat_norm, state[..., -1], state_delta)

        if was_unbatched:
            state = state.squeeze(0)

        return state

    def predict_trajectory(
        self,
        t: torch.Tensor,
        context: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(t, context, initial_state)


class RocketHybridPINNC2(nn.Module):
    """
    [PINN_V2][2025-11-19][C2 Architecture]
    Hybrid PINN with Shared Stem + Dedicated Branches architecture.

    Architecture:
    1. Shared Stem: Time + DeepContext + Temporal modeling → shared embedding
    2. Shared embedding → z0 (via Transformer encoder window)
    3. Latent ODE evolves z(t) from z0
    4. Dedicated Branches: Translation, Rotation, Mass
    5. Quaternion normalization (Set#1)
    6. Δ-state reconstruction: s = s0 + Δs

    This architecture combines:
    - Direction C (Hybrid Temporal + Latent ODE)
    - Set#1 (Output Stability: quaternion norm, Δ-state)
    - Set#2 (Deep Context Encoder)
    - C2 (Shared Stem + Dedicated Branches)
    """

    requires_initial_state = True

    def __init__(
        self,
        context_dim: int,
        latent_dim: int = 64,
        fourier_features: int = 8,
        shared_stem_hidden_dim: int = 128,
        temporal_type: str = "transformer",
        temporal_n_layers: int = 4,
        temporal_n_heads: int = 4,
        temporal_dim_feedforward: int = 512,
        encoder_window: int = 10,
        translation_branch_dims: list = None,
        rotation_branch_dims: list = None,
        mass_branch_dims: list = None,
        activation: str = "tanh",
        transformer_activation: str = "gelu",
        dynamics_n_hidden: int = 3,
        dynamics_n_neurons: int = 128,
        layer_norm: bool = True,
        dropout: float = 0.05,
        debug_stats: bool = True,
    ) -> None:
        super().__init__()

        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.encoder_window = encoder_window
        self.debug_stats_enabled = debug_stats
        self._debug_stats = {}

        # Shared Stem: Time + DeepContext + Temporal
        self.shared_stem = SharedStem(
            context_dim=context_dim,
            fourier_features=fourier_features,
            hidden_dim=shared_stem_hidden_dim,
            temporal_type=temporal_type,
            n_layers=temporal_n_layers,
            n_heads=temporal_n_heads,
            dim_feedforward=temporal_dim_feedforward,
            dropout=dropout,
            transformer_activation=transformer_activation,
        )

        # Sequence encoder to derive z0 from shared embedding
        time_dim = 1 + 2 * fourier_features
        context_embedding_dim = 32  # From DeepContextEncoder
        self.sequence_input_proj = nn.Linear(
            time_dim + context_embedding_dim, shared_stem_hidden_dim
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=shared_stem_hidden_dim,
            nhead=temporal_n_heads,
            dim_feedforward=temporal_dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=transformer_activation,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=temporal_n_layers
        )
        self.z0_proj = nn.Linear(shared_stem_hidden_dim, latent_dim)

        # Latent dynamics
        condition_dim = time_dim + context_embedding_dim
        self.dynamics_net = LatentDynamicsNet(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            n_hidden=dynamics_n_hidden,
            n_neurons=dynamics_n_neurons,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        self.ode_block = LatentODEBlock(self.dynamics_net)

        # Dedicated Branches
        self.translation_branch = TranslationBranch(
            hidden_dim=latent_dim,
            branch_dims=translation_branch_dims,
            activation="gelu",
            dropout=dropout,
        )
        self.rotation_branch = RotationBranch(
            hidden_dim=latent_dim,
            branch_dims=rotation_branch_dims,
            activation="gelu",
            dropout=dropout,
        )
        self.mass_branch = MassBranch(
            hidden_dim=latent_dim,
            branch_dims=mass_branch_dims,
            activation="gelu",
            dropout=dropout,
        )

    def _ensure_batched(
        self,
        t: torch.Tensor,
        context: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Ensure tensors are batched."""
        was_unbatched = False

        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.dim() == 2:
            t = t.unsqueeze(0)
            was_unbatched = True

        if context.dim() == 1:
            context = context.unsqueeze(0)

        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)

        return t, context, initial_state, was_unbatched

    def get_debug_stats(self, reset: bool = True) -> dict:
        """Get debug statistics from last forward pass."""
        stats = getattr(self, "_debug_stats", {})
        if reset:
            self._debug_stats = {}
        return stats

    def _capture_debug_stats(
        self,
        shared_emb: torch.Tensor,
        quat_raw: torch.Tensor,
        quat_norm: torch.Tensor,
        mass: torch.Tensor,
        delta_state: torch.Tensor,
    ) -> None:
        """Capture debug statistics."""
        if not self.debug_stats_enabled:
            return

        with torch.no_grad():
            delta_norm = torch.linalg.norm(delta_state, dim=-1)
            quat_raw_norm = torch.linalg.norm(quat_raw, dim=-1)
            quat_norm_val = torch.linalg.norm(quat_norm, dim=-1)
            if mass.shape[1] > 1:
                mass_diff = torch.diff(mass, dim=1)
                mass_violation = (mass_diff > 1e-5).float().mean().item()
            else:
                mass_violation = 0.0

            emb_stats = {
                "mean": shared_emb.mean().item(),
                "std": shared_emb.std(unbiased=False).item(),
                "min": shared_emb.min().item(),
                "max": shared_emb.max().item(),
            }

            self._debug_stats = {
                "quat_norm_raw_mean": float(quat_raw_norm.mean().item()),
                "quat_norm_raw_std": float(quat_raw_norm.std(unbiased=False).item()),
                "quat_norm_mean": float(quat_norm_val.mean().item()),
                "quat_norm_std": float(quat_norm_val.std(unbiased=False).item()),
                "mass_increase_ratio": float(mass_violation),
                "delta_state_l2_mean": float(delta_norm.mean().item()),
                "delta_state_l2_max": float(delta_norm.max().item()),
                "shared_embedding_stats": emb_stats,
            }

    def forward(
        self,
        t: torch.Tensor,
        context: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            t: Time [..., 1] or [batch, N, 1] (nondimensional)
            context: Context vector [..., context_dim] or [batch, context_dim]
            initial_state: Initial state s0 [..., 14] or [batch, 14]

        Returns:
            state: Predicted state [..., 14] or [batch, N, 14]
        """
        if initial_state is None:
            raise ValueError(
                "RocketHybridPINNC2 requires the true initial state s0."
            )

        t, context, initial_state, was_unbatched = self._ensure_batched(
            t, context, initial_state
        )
        batch_size, N, _ = t.shape

        # 1. Shared Stem: Time + DeepContext + Temporal → shared embedding
        shared_emb = self.shared_stem(t, context)  # [batch, N, hidden_dim]

        # 2. Use shared embedding to derive z0 (via Transformer encoder window)
        # Get time and context embeddings for sequence encoder
        time_dim = 1 + 2 * self.shared_stem.fourier_features
        t_emb = self.shared_stem.time_embedding(t)  # [batch, N, time_dim]
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(batch_size, N, -1)
        ctx_emb = self.shared_stem.context_encoder(context)  # [batch, N, 32]
        seq_features = torch.cat([t_emb, ctx_emb], dim=-1)

        window = min(self.encoder_window, N)
        enc_tokens = self.sequence_input_proj(seq_features[:, :window, :])
        enc_tokens = self.sequence_encoder(enc_tokens)
        z0_tokens = enc_tokens.mean(dim=1)  # [batch, hidden_dim]
        z0 = self.z0_proj(z0_tokens)  # [batch, latent_dim]

        # 3. Latent ODE integration on full grid
        condition = seq_features  # [batch, N, time_dim + 32]
        z_traj = self.ode_block(z0, t, condition)  # [batch, N, latent_dim]

        # 4. Dedicated Branches process z(t)
        translation = self.translation_branch(z_traj)  # [batch, N, 6]
        rotation_raw = self.rotation_branch(z_traj)  # [batch, N, 7]
        mass_delta = self.mass_branch(z_traj)  # [batch, N, 1]

        # 5. Quaternion normalization (Set#1)
        quat_raw = rotation_raw[..., :4]
        ang_vel = rotation_raw[..., 4:]
        quat_norm = normalize_quaternion(quat_raw)
        rotation = torch.cat([quat_norm, ang_vel], dim=-1)

        # 6. Δ-state reconstruction (Set#1)
        state_delta = torch.cat(
            [translation, rotation, mass_delta], dim=-1
        )  # [batch, N, 14]

        s0 = initial_state.unsqueeze(1).expand(-1, N, -1)  # [batch, N, 14]
        state = s0 + state_delta  # [batch, N, 14]

        # Capture debug stats
        self._capture_debug_stats(
            shared_emb, quat_raw, quat_norm, state[..., -1], state_delta
        )

        if was_unbatched:
            state = state.squeeze(0)

        return state

    def predict_trajectory(
        self,
        t: torch.Tensor,
        context: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience method for trajectory prediction."""
        return self.forward(t, context, initial_state)


class RocketHybridPINNC3(nn.Module):
    """
    [PINN_V2][2025-XX-XX][C3 Architecture]
    Enhanced Hybrid PINN with RMSE reduction solutions.
    
    C3 = C2 + 6 Architectural Solutions:
    - Solution 3: MonotonicMassBranch (structural mass constraint)
    - Solution 2: RotationBranchMinimal (quaternion minimal representation)
    - Solution 4: LatentODEBlockRK4 (higher-order integration)
    - Solution 1: PhysicsAwareTranslationBranch (explicit physics) - Phase 2
    - Solution 5: CoordinatedBranches (aerodynamic coupling) - Phase 2
    - Solution 6: EnhancedZ0Derivation (hybrid initialization) - Phase 3
    
    Architecture (Phase 1 - Core):
    1. Shared Stem: Time + DeepContext + Temporal modeling → shared embedding
    2. Shared embedding → z0 (via Transformer encoder window)
    3. RK4 Latent ODE evolves z(t) from z0
    4. Dedicated Branches: Translation, RotationMinimal, MonotonicMass
    5. Δ-state reconstruction: s = s0 + Δs
    
    Key improvements:
    - Mass always decreases (structural guarantee)
    - Quaternion always unit norm (from minimal representation)
    - Higher-order integration (RK4 vs Euler)
    """

    requires_initial_state = True

    def __init__(
        self,
        context_dim: int,
        latent_dim: int = 64,
        fourier_features: int = 8,
        shared_stem_hidden_dim: int = 128,
        temporal_type: str = "transformer",
        temporal_n_layers: int = 4,
        temporal_n_heads: int = 4,
        temporal_dim_feedforward: int = 512,
        encoder_window: int = 10,
        translation_branch_dims: list = None,
        rotation_branch_dims: list = None,
        mass_branch_dims: list = None,
        activation: str = "tanh",
        transformer_activation: str = "gelu",
        dynamics_n_hidden: int = 3,
        dynamics_n_neurons: int = 128,
        layer_norm: bool = True,
        dropout: float = 0.05,
        debug_stats: bool = True,
        use_physics_aware_translation: bool = False,
        use_coordinated_branches: bool = False,
    ) -> None:
        super().__init__()

        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.encoder_window = encoder_window
        self.debug_stats_enabled = debug_stats
        self._debug_stats = {}

        # Shared Stem: Time + DeepContext + Temporal
        self.shared_stem = SharedStem(
            context_dim=context_dim,
            fourier_features=fourier_features,
            hidden_dim=shared_stem_hidden_dim,
            temporal_type=temporal_type,
            n_layers=temporal_n_layers,
            n_heads=temporal_n_heads,
            dim_feedforward=temporal_dim_feedforward,
            dropout=dropout,
            transformer_activation=transformer_activation,
        )

        # Sequence encoder to derive z0 from shared embedding
        time_dim = 1 + 2 * fourier_features
        context_embedding_dim = 32  # From DeepContextEncoder
        self.sequence_input_proj = nn.Linear(
            time_dim + context_embedding_dim, shared_stem_hidden_dim
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=shared_stem_hidden_dim,
            nhead=temporal_n_heads,
            dim_feedforward=temporal_dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=transformer_activation,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=temporal_n_layers
        )
        self.z0_proj = nn.Linear(shared_stem_hidden_dim, latent_dim)

        # Latent dynamics
        condition_dim = time_dim + context_embedding_dim
        self.dynamics_net = LatentDynamicsNet(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            n_hidden=dynamics_n_hidden,
            n_neurons=dynamics_n_neurons,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        # Solution 4: RK4 integration instead of Euler
        self.ode_block = LatentODEBlockRK4(self.dynamics_net)

        # Physics computation layer (for Phase 2 features)
        self.physics_layer = PhysicsComputationLayer()

        # Dedicated Branches
        # Solution 2: RotationBranchMinimal (quaternion minimal representation)
        rotation_branch = RotationBranchMinimal(
            hidden_dim=latent_dim,
            branch_dims=rotation_branch_dims,
            activation="gelu",
            dropout=dropout,
        )
        # Solution 3: MonotonicMassBranch (structural mass constraint)
        mass_branch = MonotonicMassBranch(
            hidden_dim=latent_dim,
            branch_dims=mass_branch_dims,
            activation="gelu",
            dropout=dropout,
        )

        # Translation branch: Phase 1 (standard) or Phase 2 (physics-aware)
        if use_physics_aware_translation:
            translation_branch = PhysicsAwareTranslationBranch(
                hidden_dim=latent_dim,
                branch_dims=translation_branch_dims,
                activation="gelu",
                dropout=dropout,
                physics_layer=self.physics_layer,
            )
        else:
            translation_branch = TranslationBranch(
                hidden_dim=latent_dim,
                branch_dims=translation_branch_dims,
                activation="gelu",
                dropout=dropout,
            )

        # Branch coordination: Phase 2 feature
        if use_coordinated_branches:
            self.branches = CoordinatedBranches(
                translation_branch=translation_branch,
                rotation_branch=rotation_branch,
                mass_branch=mass_branch,
                physics_layer=self.physics_layer,
            )
            self.use_coordination = True
        else:
            # Direct branch access (Phase 1)
            self.translation_branch = translation_branch
            self.rotation_branch = rotation_branch
            self.mass_branch = mass_branch
            self.use_coordination = False

    def _ensure_batched(
        self,
        t: torch.Tensor,
        context: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Ensure tensors are batched."""
        was_unbatched = False

        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.dim() == 2:
            t = t.unsqueeze(0)
            was_unbatched = True

        if context.dim() == 1:
            context = context.unsqueeze(0)

        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)

        return t, context, initial_state, was_unbatched

    def get_debug_stats(self, reset: bool = True) -> dict:
        """Get debug statistics from last forward pass."""
        stats = getattr(self, "_debug_stats", {})
        if reset:
            self._debug_stats = {}
        return stats

    def _capture_debug_stats(
        self,
        shared_emb: torch.Tensor,
        quaternion: torch.Tensor,
        mass: torch.Tensor,
        delta_state: torch.Tensor,
    ) -> None:
        """Capture debug statistics."""
        if not self.debug_stats_enabled:
            return

        with torch.no_grad():
            delta_norm = torch.linalg.norm(delta_state, dim=-1)
            quat_norm_val = torch.linalg.norm(quaternion, dim=-1)
            if mass.shape[1] > 1:
                mass_diff = torch.diff(mass, dim=1)
                mass_violation = (mass_diff > 1e-5).float().mean().item()
            else:
                mass_violation = 0.0

            emb_stats = {
                "mean": shared_emb.mean().item(),
                "std": shared_emb.std(unbiased=False).item(),
                "min": shared_emb.min().item(),
                "max": shared_emb.max().item(),
            }

            self._debug_stats = {
                "quat_norm_mean": float(quat_norm_val.mean().item()),
                "quat_norm_std": float(quat_norm_val.std(unbiased=False).item()),
                "mass_increase_ratio": float(mass_violation),  # Should be 0.0 for C3
                "delta_state_l2_mean": float(delta_norm.mean().item()),
                "delta_state_l2_max": float(delta_norm.max().item()),
                "shared_embedding_stats": emb_stats,
            }

    def forward(
        self,
        t: torch.Tensor,
        context: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            t: Time [..., 1] or [batch, N, 1] (nondimensional)
            context: Context vector [..., context_dim] or [batch, context_dim]
            initial_state: Initial state s0 [..., 14] or [batch, 14]

        Returns:
            state: Predicted state [..., 14] or [batch, N, 14]
        """
        if initial_state is None:
            raise ValueError(
                "RocketHybridPINNC3 requires the true initial state s0."
            )

        t, context, initial_state, was_unbatched = self._ensure_batched(
            t, context, initial_state
        )
        batch_size, N, _ = t.shape

        # 1. Shared Stem: Time + DeepContext + Temporal → shared embedding
        shared_emb = self.shared_stem(t, context)  # [batch, N, hidden_dim]

        # 2. Use shared embedding to derive z0 (via Transformer encoder window)
        # Get time and context embeddings for sequence encoder
        time_dim = 1 + 2 * self.shared_stem.fourier_features
        t_emb = self.shared_stem.time_embedding(t)  # [batch, N, time_dim]
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(batch_size, N, -1)
        ctx_emb = self.shared_stem.context_encoder(context)  # [batch, N, 32]
        seq_features = torch.cat([t_emb, ctx_emb], dim=-1)

        window = min(self.encoder_window, N)
        enc_tokens = self.sequence_input_proj(seq_features[:, :window, :])
        enc_tokens = self.sequence_encoder(enc_tokens)
        z0_tokens = enc_tokens.mean(dim=1)  # [batch, hidden_dim]
        z0 = self.z0_proj(z0_tokens)  # [batch, latent_dim]

        # 3. RK4 Latent ODE integration on full grid (Solution 4)
        condition = seq_features  # [batch, N, time_dim + 32]
        z_traj = self.ode_block(z0, t, condition)  # [batch, N, latent_dim]

        # 4. Dedicated Branches process z(t)
        # Extract initial mass from initial_state
        m0 = initial_state[..., -1:]  # [batch, 1]

        if self.use_coordination:
            # Phase 2: Coordinated branches with aerodynamic coupling
            translation, rotation, mass = self.branches(z_traj, m0)
        else:
            # Phase 1: Direct branch access
            rotation = self.rotation_branch(z_traj)  # [batch, N, 7] (quaternion already unit norm)
            mass = self.mass_branch(z_traj, m0)  # [batch, N, 1] (always decreasing)
            
            # Translation branch (with optional physics-aware correction)
            if isinstance(self.translation_branch, PhysicsAwareTranslationBranch):
                translation = self.translation_branch(z_traj, mass)  # [batch, N, 6]
            else:
                translation = self.translation_branch(z_traj)  # [batch, N, 6]

        # 5. No quaternion normalization needed (Solution 2: handled by minimal representation)
        # rotation already contains unit quaternion [q0, q1, q2, q3, wx, wy, wz]

        # 6. Δ-state reconstruction (Set#1)
        state_delta = torch.cat(
            [translation, rotation, mass], dim=-1
        )  # [batch, N, 14]

        s0 = initial_state.unsqueeze(1).expand(-1, N, -1)  # [batch, N, 14]
        state = s0 + state_delta  # [batch, N, 14]

        # Capture debug stats
        quaternion = rotation[..., :4]  # Extract quaternion for stats
        self._capture_debug_stats(
            shared_emb, quaternion, state[..., -1:], state_delta
        )

        if was_unbatched:
            state = state.squeeze(0)

        return state

    def predict_trajectory(
        self,
        t: torch.Tensor,
        context: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience method for trajectory prediction."""
        return self.forward(t, context, initial_state)


