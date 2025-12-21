import torch
import torch.nn as nn

from ..core import NormalizationLayer, DropoutLayer

from ...config.components.composite import EncoderConfig, TransformerEncoderConfig
from ...config.components.core import NormalizationLayerConfig, DropoutLayerConfig
from ...utils import get_block, get_activation

class Encoder(nn.Module):
    """A modular encoder that builds a sequence of layers based on the provided configuration.

    Args:
        config (EncoderConfig): Configuration for the encoder.
    """
    def __init__(self, config: EncoderConfig):
        super(Encoder, self).__init__()

        self.layer_configs = config.layer_configs
        self.final_activation = config.final_activation
        self.normalization = config.normalization
        self.activation = config.activation
        self.dropout_p = config.dropout_p
        self.final_activation = config.final_activation

        # Build the main generator body using the same modular layer system as Decoder
        layers = nn.ModuleList()

        for layer_config in self.layer_configs:
            layer_config.normalization = self.normalization
            layer_config.activation = self.activation
            layer_config.dropout_p = self.dropout_p
            layers.append(get_block(layer_config))

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
        self.normalization = config.normalization
        self.activation = config.activation
        self.dropout_p = config.dropout_p
        self.final_activation = config.final_activation

        # Transformer Encoder stack (self-attention only)
        self.layers = nn.ModuleList([])
        for _ in range(self.num_layers):
            layer = nn.Sequential(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.hidden_dim * 4,
                    dropout=self.dropout_p,
                    batch_first=True,
                    activation=self.activation
                ),
                NormalizationLayer(
                    config=NormalizationLayerConfig(
                        norm_type=self.normalization,
                        num_features=self.hidden_dim
                    )
                )
            )          

        self.final_act = get_activation(self.final_activation)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the Transformer encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim).
        """

        # Pass through all encoder layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attn_mask)

        # Optional activation
        x = self.final_act(x)
        return x