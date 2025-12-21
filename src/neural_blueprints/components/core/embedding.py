import torch
import torch.nn as nn

from ...config.components.core import EmbeddingLayerConfig, PatchEmbeddingLayerConfig, NormalizationLayerConfig, DropoutLayerConfig
from ...utils import get_activation

import logging
logger = logging.getLogger(__name__)

class EmbeddingLayer(nn.Module):
    """Embedding layer component.
    
    Args:
        config (EmbeddingLayerConfig): Configuration for the embedding layer.
    """
    def __init__(self, config: EmbeddingLayerConfig):
        super(EmbeddingLayer, self).__init__()
        from ..core import NormalizationLayer, DropoutLayer

        self.embedding_layer = nn.Embedding(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
            padding_idx=config.padding_idx
        )
        self.normalization_layer = NormalizationLayer(
            config = NormalizationLayerConfig(
                norm_type=config.normalization,
                num_features=config.embedding_dim
            )
        )
        self.activation_layer = get_activation(config.activation)
        self.dropout_layer = DropoutLayer(
            config = DropoutLayerConfig(
                p=config.dropout_p
            )
        )

        self.layer = nn.Sequential(
            self.embedding_layer,
            self.normalization_layer,
            self.activation_layer,
            self.dropout_layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the embedding layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
        
        Returns:
            Output tensor of shape (batch_size, embedding_dim).
        """
        try:
            x = x.long()
            z = self.embedding_layer(x)    # shape (batch_size, sequence_length, embedding_dim)
        except Exception as e:
            logger.debug(f"Input tensor x: {x}")
            logger.debug(f"Input tensor x dtype: {x.dtype}")
            logger.debug(f"Input tensor x shape: {x.shape}")
            logger.debug(f"Input tensor x unique values: {torch.unique(x, sorted=True)}")
            logger.debug(f"Embedding layer config: num_embeddings={self.embedding_layer.num_embeddings}, embedding_dim={self.embedding_layer.embedding_dim}, padding_idx={self.embedding_layer.padding_idx}")
            logger.error(f"Error in EmbeddingLayer forward pass: {e}")
            raise e
        z = self.normalization_layer(z)
        z = self.activation_layer(z)
        return self.dropout_layer(z)
    
class PatchEmbeddingLayer(nn.Module):
    """Patch Embedding layer component.
    
    Args:
        config (PatchEmbeddingLayerConfig): Configuration for the patch embedding layer.
    """
    def __init__(self, config: PatchEmbeddingLayerConfig):
        super(PatchEmbeddingLayer, self).__init__()
        from ..core import NormalizationLayer, DropoutLayer

        self.patch_size = config.patch_size
        self.embedding_dim = config.embedding_dim
        self.in_channels = config.in_channels

        embedding = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embedding_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        normalization = NormalizationLayer(
            config = NormalizationLayerConfig(
                norm_type=config.normalization,
                normalized_dim=config.embedding_dim
            )
        )
        activation = get_activation(config.activation)

        self.patch_embedding_layer = nn.Sequential(
            embedding,
            normalization,
            activation
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the patch embedding layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            Output tensor of shape (batch_size, num_patches, embedding_dim).
        """
        x = self.patch_embedding_layer(x)  # (batch_size, embedding_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)      # (batch_size, embedding_dim, num_patches)
        x = x.transpose(1, 2) # (batch_size, num_patches, embedding_dim)
        return x
