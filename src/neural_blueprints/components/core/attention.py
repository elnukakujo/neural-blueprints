import torch
import torch.nn as nn
from ...config import AttentionLayerConfig

class AttentionLayer(nn.Module):
    """A multi-head attention layer.

    Args:
        config (AttentionLayerConfig): Configuration for the attention layer.
    """
    def __init__(self, config: AttentionLayerConfig):
        super(AttentionLayer, self).__init__()
        self.input_dim = config.input_dim
        self.num_heads = config.num_heads
        
        self.attention = nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=self.num_heads)
        self.norm = nn.LayerNorm(self.input_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Forward pass through the attention layer.

        Args:
            query (torch.Tensor): Query tensor of shape (seq_len, batch_size, input_dim).
            key (torch.Tensor): Key tensor of shape (seq_len, batch_size, input_dim).
            value (torch.Tensor): Value tensor of shape (seq_len, batch_size, input_dim).
        
        Returns:
            Output tensor of shape (seq_len, batch_size, input_dim).
        """
        attn_output, _ = self.attention(query, key, value)
        out = self.norm(query + attn_output)
        return out