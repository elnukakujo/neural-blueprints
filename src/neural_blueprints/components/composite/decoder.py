import torch
import torch.nn as nn

from ...config.components.composite import DecoderConfig, TransformerDecoderConfig
from ...utils import get_block, get_activation

class Decoder(nn.Module):
    """A modular decoder that builds a sequence of layers based on the provided configuration.

    Args:
        config (DecoderConfig): Configuration for the decoder.
    """
    def __init__(self, config: DecoderConfig):
        super(Decoder, self).__init__()

        self.normalization = config.normalization
        self.activation = config.activation
        self.dropout_p = config.dropout_p
        self.layer_configs = config.layer_configs
        self.final_activation = config.final_activation

        # Build the main generator body using the same modular layer system as Decoder
        layers = nn.ModuleList()

        for layer_config in self.layer_configs:
            layer_config.normalization = self.normalization if layer_config.normalization is None else layer_config.normalization
            layer_config.activation = self.activation if layer_config.activation is None else layer_config.activation
            layer_config.dropout_p = self.dropout_p if layer_config.dropout_p is None else layer_config.dropout_p
            layers.append(get_block(layer_config))

        layers.append(get_activation(self.final_activation))

        self.network = nn.Sequential(*layers)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            Output tensor after passing through the decoder.
        """
        return self.network(x)
    
class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerDecoderConfig):
        """Initialize the TransformerDecoder.

        Args:
            config (TransformerDecoderConfig): Configuration for the transformer decoder.
        """
        super().__init__()
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.dropout = config.dropout_p
        self.final_activation = config.final_activation

        # Optional input projection
        self.input_proj = (
            nn.Linear(self.input_dim, self.hidden_dim)
            if self.input_dim != self.hidden_dim
            else nn.Identity()
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=self.dropout,
                activation="relu"
            )
            for _ in range(self.num_layers)
        ])

        # Optional normalization + activation
        self.final_act = get_activation(self.final_activation)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            memory (torch.Tensor): Memory tensor from the encoder of shape (batch_size, mem_seq_len, hidden_dim).

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        x = self.input_proj(x)

        # Transpose to (seq_len, batch_size, hidden_dim)
        x = x.transpose(0, 1)
        memory = memory.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, memory)

        # Back to (batch_size, seq_len, hidden_dim)
        x = x.transpose(0, 1)
        x = self.final_act(x)
        return x