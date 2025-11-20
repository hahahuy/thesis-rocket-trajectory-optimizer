"""
Neural network architectures for PINN models.

Provides MLP blocks, Fourier feature embeddings, and layer normalization utilities.
"""

import torch
import torch.nn as nn
from typing import Optional, Sequence


class MLPBlock(nn.Module):
    """Standard MLP block with optional layer normalization and dropout."""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "tanh",
        layer_norm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.linear = nn.Linear(in_dim, out_dim)
        
        # Activation
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "identity":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class FourierFeatures(nn.Module):
    """
    Fourier feature embedding for time input.
    
    Maps time t to [t, sin(2πk t), cos(2πk t)] for k = 1..K.
    """
    
    def __init__(self, n_frequencies: int = 8):
        super().__init__()
        self.n_frequencies = n_frequencies
        # Frequencies: k = 1, 2, ..., n_frequencies
        self.register_buffer(
            'frequencies',
            torch.arange(1, n_frequencies + 1, dtype=torch.float32)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time tensor [..., 1] (nondimensional)
            
        Returns:
            features: Fourier features [..., 1 + 2*n_frequencies]
        """
        # t itself
        features = [t]
        
        # Sin and cos for each frequency
        for k in self.frequencies:
            angle = 2 * torch.pi * k * t
            features.append(torch.sin(angle))
            features.append(torch.cos(angle))
        
        return torch.cat(features, dim=-1)


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable depth and width.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = None,
        activation: str = "tanh",
        layer_norm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128] * 6
        
        dims = [input_dim] + hidden_dims + [output_dim]
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(
                MLPBlock(
                    dims[i],
                    dims[i + 1],
                    activation=activation if i < len(dims) - 2 else "identity",
                    layer_norm=layer_norm if i < len(dims) - 2 else False,
                    dropout=dropout if i < len(dims) - 2 else 0.0
                )
            )
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TimeEmbedding(nn.Module):
    """
    Time embedding combining raw time and Fourier features.
    """
    
    def __init__(self, n_frequencies: int = 8):
        super().__init__()
        self.fourier = FourierFeatures(n_frequencies)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time [..., 1] (nondimensional)
            
        Returns:
            embedding: [..., 1 + 2*n_frequencies]
        """
        return self.fourier(t)


class ContextEncoder(nn.Module):
    """
    Encodes context parameters into a fixed-size embedding.
    """
    
    def __init__(
        self,
        context_dim: int,
        embedding_dim: int = 16,
        activation: str = "tanh"
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, embedding_dim),
            nn.Tanh() if activation == "tanh" else nn.ReLU()
        )
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: Context vector [..., context_dim]
            
        Returns:
            embedding: [..., embedding_dim]
        """
        return self.encoder(context)


class DeepContextEncoder(nn.Module):
    """
    Deeper context encoder with multiple layers, GELU activations, and LayerNorm.
    
    Default shape: context_dim → 64 → 128 → 128 → 64 → 32
    """

    def __init__(
        self,
        context_dim: int,
        hidden_dims: Sequence[int] = (64, 128, 128, 64),
        output_dim: int = 32,
    ) -> None:
        super().__init__()

        layers = []
        in_dim = context_dim
        for idx, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.GELU())
            # LayerNorm on early layers for stability (match spec)
            if idx < 2:
                layers.append(nn.LayerNorm(dim))
            in_dim = dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class OutputHeads(nn.Module):
    """
    Split output heads for translation, rotation, and mass predictions.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.translation_head = nn.Linear(in_dim, 6)
        self.rotation_head = nn.Linear(in_dim, 7)
        self.mass_head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor):
        translation = self.translation_head(x)
        rotation = self.rotation_head(x)
        mass = self.mass_head(x)
        return translation, rotation, mass


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize quaternion to unit length with numerical stability.
    """

    norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
    return q / norm

