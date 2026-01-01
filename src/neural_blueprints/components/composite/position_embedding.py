import torch
import torch.nn as nn

from .base import BaseComposite
from ..core import EmbeddingLayer

from ...config.components.composite import PositionEmbeddingConfig
from ...config.components.core import EmbeddingLayerConfig

class PositionEmbedding(BaseComposite):
    """Positional Embedding Module.
    
    This module generates positional embeddings for input sequences, allowing the model to
    incorporate information about the position of each element in the sequence.

    Args:
        config (PositionEmbeddingConfig): Configuration for the Position Embedding module.
    """
    def __init__(self,
                    config: PositionEmbeddingConfig
                ):
        super().__init__()
        
        self.input_dim = [config.num_positions]
        self.output_dim = [config.num_positions, config.latent_dim]

        normalization = config.normalization
        activation = config.activation
        dropout_p = config.dropout_p

        self.position_embedding = EmbeddingLayer(
            config=EmbeddingLayerConfig(
                num_embeddings=self.input_dim[0],
                embedding_dim=self.output_dim[-1],
                padding_idx=None,
                normalization=normalization,
                activation=activation,
                dropout_p=dropout_p
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate positional embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Positional embeddings of shape (batch_size, seq_len, output_dim).
        """
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), x.size(1)) # shape: (batch, seq)
        try:
            return self.position_embedding(pos)  # shape: (batch, seq, output_dim)
        except Exception as e:
            assert pos.size(1) == self.input_dim[0], f"Input sequence length {pos.size(1)} does not match configured num_positions {self.input_dim[0]}"
            raise e