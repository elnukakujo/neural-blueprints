"""Vision Transformer (ViT)"""
import torch
import torch.nn as nn
from typing import Tuple
from ..base import NeuralNetworkBase
from .transformer_base import TransformerBlock, PositionalEncoding
import math


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""
    
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, height, width)
        x = self.projection(x)  # (batch, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x


class VisionTransformer(NeuralNetworkBase):
    """
    Vision Transformer for image classification.
    Converts images to patches and processes with transformer encoder.
    """
    
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, num_classes: int = 1000,
                 d_model: int = 768, num_heads: int = 12, num_layers: int = 12,
                 d_ff: int = 3072, dropout: float = 0.1):
        """
        Args:
            image_size: Input image size (assumed square)
            patch_size: Size of each patch
            in_channels: Number of input channels
            num_classes: Number of output classes
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, d_model)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, is_decoder=False)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images (batch, channels, height, width)
            
        Returns:
            Class logits (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Patch embedding
        x = self.patch_embedding(x)  # (batch, num_patches, d_model)
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches + 1, d_model)
        
        # Add positional embedding
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Classification from CLS token
        x = self.norm(x)
        cls_output = x[:, 0]  # Take CLS token
        logits = self.head(cls_output)
        
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head"""
        batch_size = x.size(0)
        
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        return x[:, 0]  # Return CLS token features