import torch
import torch.nn as nn

from ...utils import get_activation
from ...config.components.core import AttentionLayerConfig, NormalizationLayerConfig, DropoutLayerConfig

class AttentionLayer(nn.Module):
    """A multi-head attention layer.

    Args:
        config (AttentionLayerConfig): Configuration for the attention layer.
    """
    def __init__(self, config: AttentionLayerConfig):
        super(AttentionLayer, self).__init__()
        from ..core import NormalizationLayer, DropoutLayer
        
        self.input_dim = config.input_dim
        self.num_heads = config.num_heads
        
        self.attention = nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=self.num_heads)
        self.normalization_layer = NormalizationLayer(
            config=NormalizationLayerConfig(
                norm_type=config.normalization,
                num_features=self.input_dim
            )
        )
        self.dropout_layer = DropoutLayer(
            config=DropoutLayerConfig(
                p=config.dropout_p
            )
        )
        self.activation_layer = get_activation(config.activation)

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
        output = query + attn_output
        output = self.normalization_layer(output)
        output = self.activation_layer(output)
        return self.dropout_layer(output)