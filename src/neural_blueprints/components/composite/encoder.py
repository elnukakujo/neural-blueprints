from neural_blueprints.utils.normalization import get_normalization
import torch
import torch.nn as nn

from ..core import ProjectionLayer
from ...config import EncoderConfig, TransformerEncoderConfig
from ...utils import get_block, get_activation

class Encoder(nn.Module):
    """A modular encoder that builds a sequence of layers based on the provided configuration.

    Args:
        config (EncoderConfig): Configuration for the encoder.
    """
    def __init__(self, config: EncoderConfig):
        super(Encoder, self).__init__()

        self.layer_types = config.layer_types
        self.layer_configs = config.layer_configs
        self.projection = config.projection
        self.final_activation = config.final_activation

        # Build the main generator body using the same modular layer system as Decoder
        layers = nn.ModuleList()

        # Optional linear projection from latent space to the first hidden dimension
        if self.projection is not None:
            layers.append(ProjectionLayer(self.projection))

        for layer_type, layer_config in zip(self.layer_types, self.layer_configs):
            layers.append(get_block(layer_type, layer_config))

        layers.append(get_activation(self.final_activation))

        self.network = nn.Sequential(*layers)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Output tensor after passing through the encoder.
        """
        return self.network(x)
    
class TransformerEncoder(nn.Module):
    """
    A clean Transformer Encoder block suitable for:

    - Generic Transformer encoder
    - Encoder-decoder transformers (as the encoder)
    - BERT-style models (encoder-only)

    Args:
        config (TransformerEncoderConfig): Configuration for the Transformer Encoder.
    """
    def __init__(self, config: TransformerEncoderConfig):
        super().__init__()

        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.dropout = config.dropout

        # Optional input projection (input_dim → hidden_dim)
        self.input_proj = (
            ProjectionLayer(config.projection)
            if config.projection is not None
            else nn.Identity()
        )

        # Transformer Encoder stack (self-attention only)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=self.dropout,
                batch_first=True,
                activation="relu"
            )
            for _ in range(self.num_layers)
        ])

        # Optional final normalizing block
        self.final_norm = get_normalization(config.final_normalization)
        self.final_act = get_activation(config.final_activation)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the Transformer encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        # Optionally project to hidden dimension
        x = self.input_proj(x)

        # Pass through all encoder layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attn_mask)

        # Optional normalization + activation
        x = self.final_norm(x)
        x = self.final_act(x)
        return x