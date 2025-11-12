import torch
import torch.nn as nn
from ...config import EmbeddingLayerConfig, PatchEmbeddingLayerConfig

class EmbeddingLayer(nn.Module):
    """Embedding layer component.
    
    Args:
        config (EmbeddingLayerConfig): Configuration for the embedding layer.
    """
    def __init__(self, config: EmbeddingLayerConfig):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the embedding layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        return self.embedding(x)
    
class PatchEmbeddingLayer(nn.Module):
    """Patch Embedding layer component.
    
    Args:
        config (EmbeddingLayerConfig): Configuration for the patch embedding layer.
        patch_size (int): Size of each patch.
        in_channels (int): Number of input channels.
    """
    def __init__(self, config: PatchEmbeddingLayerConfig):
        super(PatchEmbeddingLayer, self).__init__()
        self.patch_size = config.patch_size
        self.embedding_dim = config.embedding_dim
        self.in_channels = config.in_channels
        self.embedding = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embedding_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the patch embedding layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, embedding_dim).
        """
        x = self.embedding(x)  # (batch_size, embedding_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)      # (batch_size, embedding_dim, num_patches)
        x = x.transpose(1, 2) # (batch_size, num_patches, embedding_dim)
        return x
