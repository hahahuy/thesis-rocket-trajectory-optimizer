"""
[PINN_V2][2025-11-19][C2 Architecture]
Shared Stem module for unified temporal + context processing.

This module implements the shared stem that processes time and context
to produce a unified embedding used by all dedicated branches.
"""

from typing import Tuple

import torch
import torch.nn as nn

from .architectures import DeepContextEncoder, FourierFeatures


class SharedStem(nn.Module):
    """
    [PINN_V2][2025-11-19][C2 Architecture]
    Shared stem that processes time and context to produce unified embedding.
    
    Architecture:
    1. Time encoding (FourierFeatures)
    2. Context encoding (DeepContextEncoder from Set#2)
    3. Temporal modeling (TransformerEncoder)
    4. Output: Shared embedding [batch, N, hidden_dim]
    """

    def __init__(
        self,
        context_dim: int,
        fourier_features: int = 8,
        hidden_dim: int = 128,
        temporal_type: str = "transformer",
        n_layers: int = 4,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.05,
        transformer_activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.context_dim = context_dim
        self.fourier_features = fourier_features
        self.hidden_dim = hidden_dim
        self.temporal_type = temporal_type

        # Time encoding: t -> [t, sin(2πk t), cos(2πk t)] for k=1..K
        self.time_embedding = FourierFeatures(n_frequencies=fourier_features)
        time_dim = 1 + 2 * fourier_features  # 17 for fourier_features=8

        # Context encoding: DeepContextEncoder from Set#2
        # Default: context_dim -> 64 -> 128 -> 128 -> 64 -> 32
        self.context_encoder = DeepContextEncoder(
            context_dim=context_dim,
            hidden_dims=(64, 128, 128, 64),
            output_dim=32,
        )
        context_embedding_dim = 32

        # Project combined features to hidden_dim
        self.input_proj = nn.Linear(
            time_dim + context_embedding_dim, hidden_dim
        )

        # Temporal module
        if temporal_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation=transformer_activation,
            )
            self.temporal = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_layers
            )
        else:
            raise NotImplementedError(
                f"Temporal type '{temporal_type}' not implemented. "
                "Supported: 'transformer'"
            )

    def _ensure_batched(
        self, t: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Ensure tensors are batched for processing.
        Returns possibly expanded tensors and whether original input was unbatched.
        """
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
        Forward pass producing shared embedding.

        Args:
            t: Time [..., 1] or [batch, N, 1] (nondimensional)
            context: Context vector [..., context_dim] or [batch, context_dim]

        Returns:
            shared_embedding: [..., hidden_dim] or [batch, N, hidden_dim]
        """
        t, context, was_unbatched = self._ensure_batched(t, context)
        batch_size, N, _ = t.shape

        # (1) Time feature embedding
        t_emb = self.time_embedding(t)  # [batch, N, time_dim]

        # (2) Broadcast context + deep encode
        if context.dim() == 2:
            # context: [batch, context_dim] -> [batch, N, context_dim]
            context = context.unsqueeze(1).expand(batch_size, N, -1)
        elif context.dim() == 3 and context.shape[1] == 1:
            # context: [batch, 1, context_dim] -> [batch, N, context_dim]
            context = context.expand(batch_size, N, -1)

        ctx_emb = self.context_encoder(context)  # [batch, N, 32]

        # (3) Combine time and context embeddings
        x = torch.cat([t_emb, ctx_emb], dim=-1)  # [batch, N, time_dim + 32]
        x = self.input_proj(x)  # [batch, N, hidden_dim]

        # (4) Temporal processing
        x = self.temporal(x)  # [batch, N, hidden_dim]

        # Remove batch dimension if inputs were unbatched
        if was_unbatched:
            x = x.squeeze(0)  # [N, hidden_dim]

        return x




