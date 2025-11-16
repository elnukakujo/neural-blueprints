from neural_blueprints.utils.normalization import get_normalization
import torch
import torch.nn as nn

from ..core import ProjectionLayer
from ...config import EncoderConfig, TransformerEncoderConfig
from ...utils import get_block, get_activation

class Encoder(nn.Module):
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
        return self.network(x)
    
class TransformerEncoder(nn.Module):
    """
    A clean Transformer Encoder block suitable for:
    - Generic Transformer encoder
    - Encoder-decoder transformers (as the encoder)
    - BERT-style models (encoder-only)

    Args:
        config (TransformerEncoderConfig): Configuration for the Transformer Encoder.
            - input_dim: Dimension of the input features.
            - hidden_dim: Dimension of the hidden features.
            - num_layers: Number of Transformer encoder layers.
            - num_heads: Number of attention heads.
            - dropout: Dropout rate for the layers.
            - projection: Optional projection layer configuration to map input_dim to hidden_dim.
            - final_normalization: Optional normalization configuration for the final output.
            - final_activation: Optional activation function for the final output.
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
        """
        x: (batch, seq, input_dim)
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