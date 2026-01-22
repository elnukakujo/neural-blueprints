import torch
import torch.nn as nn

from .base import BaseComposite
from ...config.components.composite import EncoderConfig, TransformerEncoderConfig
from ...config.components.core import DenseLayerConfig
from ...utils import get_block, get_activation

import logging
logger = logging.getLogger(__name__)

class Encoder(BaseComposite):
    """A modular encoder that builds a sequence of layers based on the provided configuration.

    Args:
        config (EncoderConfig): Configuration for the encoder.
    """
    def __init__(self, config: EncoderConfig):
        super(Encoder, self).__init__()

        self.input_dim = config.layers_dim[0]
        self.output_dim = config.latent_dim

        layers_dim = config.layers_dim + [config.latent_dim]

        normalization = config.normalization
        activation = config.activation
        dropout_p = config.dropout_p
        final_activation = config.final_activation

        # Build the main generator body using the same modular layer system as Decoder
        layers = nn.ModuleList()

        for layer_idx in range(len(layers_dim)-1):
            layers.append(get_block(
                DenseLayerConfig(
                    input_dim=[layers_dim[layer_idx]],
                    output_dim=[layers_dim[layer_idx + 1]],
                    normalization=normalization,
                    activation=activation if layer_idx != len(layers_dim)-2 else None,
                    dropout_p=dropout_p
                )
            ))

        layers.append(get_activation(final_activation))

        self.network = nn.Sequential(*layers)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Output tensor after passing through the encoder.
        """
        return self.network(x)
    
class TransformerEncoder(BaseComposite):
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
        self.output_dim = config.input_dim

        dim_feedforward = config.dim_feedforward        
        num_layers = config.num_layers
        num_heads = config.num_heads
        activation = config.activation
        dropout_p = config.dropout_p
        final_activation = config.final_activation


        # Transformer Encoder stack (self-attention only)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=self.input_dim[-1],
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout_p,
                    batch_first=True,
                    activation=activation
                )
            )

        self.final_act = get_activation(final_activation)
        
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