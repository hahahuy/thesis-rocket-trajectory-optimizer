"""
[PINN_V2][2025-11-18][Direction B]
Sequence-based PINN architecture using a Transformer encoder.

Implements Direction B from `docs/PINN_v2_Refactor_Directions.md`.
Treats the full time grid as a sequence and models temporal
dependencies via a Transformer encoder before predicting the 14-D state.
"""

from typing import Tuple

import torch
import torch.nn as nn

from .architectures import ContextEncoder, MLP, TimeEmbedding


class RocketSequencePINN(nn.Module):
    """
    [PINN_V2][2025-11-18][Direction B]
    Sequence model (Transformer) for predicting rocket trajectories.

    Input: (t, context) -> Output: state trajectory [batch, N, 14]
    """

    def __init__(
        self,
        context_dim: int,
        context_embedding_dim: int = 64,
        fourier_features: int = 8,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.05,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.context_dim = context_dim
        self.fourier_features = fourier_features
        self.context_embedding_dim = context_embedding_dim

        # Embeddings
        self.time_embedding = TimeEmbedding(n_frequencies=fourier_features)
        time_dim = 1 + 2 * fourier_features

        self.context_encoder = ContextEncoder(
            context_dim=context_dim,
            embedding_dim=context_embedding_dim,
            activation="tanh",
        )

        # Project embeddings to Transformer dimension
        self.input_proj = nn.Linear(time_dim + context_embedding_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=activation,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )

        # Output head -> 14-D state
        self.output_head = MLP(
            input_dim=d_model,
            output_dim=14,
            hidden_dims=[d_model, d_model],
            activation="tanh",
            layer_norm=True,
            dropout=dropout,
        )

    def _ensure_batched(
        self, t: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Ensure tensors are batched for sequence processing.
        Returns possibly expanded tensors and whether original input was unbatched.
        """
        t_was_unbatched = False
        context_was_unbatched = False

        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [N, 1]
        if t.dim() == 2:
            t = t.unsqueeze(0)
            t_was_unbatched = True

        if context.dim() == 1:
            context = context.unsqueeze(0)
            context_was_unbatched = True

        return t, context, t_was_unbatched and context_was_unbatched

    def forward(self, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing predicted trajectory.

        Args:
            t: [batch, N, 1] time grid (or compatible unbatched shapes)
            context: [batch, context_dim]
        """
        t, context, was_unbatched = self._ensure_batched(t, context)

        batch_size, N, _ = t.shape

        # Broadcast context to time dimension
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(batch_size, N, -1)

        # Embeddings
        t_emb = self.time_embedding(t)
        ctx_emb = self.context_encoder(context)

        seq = torch.cat([t_emb, ctx_emb], dim=-1)
        seq = self.input_proj(seq)

        seq = self.transformer(seq)
        state = self.output_head(seq)

        if was_unbatched:
            state = state.squeeze(0)

        return state

    def predict_trajectory(
        self, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Convenience helper mirroring other architectures."""
        return self.forward(t, context)


