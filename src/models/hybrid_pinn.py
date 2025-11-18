"""
[PINN_V2][2025-11-18][Direction C]
Hybrid PINN combining a Transformer-based encoder for the latent
initial state with latent ODE dynamics.
"""

from typing import Tuple

import torch
import torch.nn as nn

from .architectures import ContextEncoder, MLP, TimeEmbedding
from .latent_ode import LatentDynamicsNet, LatentODEBlock


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


