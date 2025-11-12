import torch
import torch.nn as nn

from ...config import DecoderConfig, TransformerDecoderConfig
from ...utils import get_block, get_activation

class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super(Decoder, self).__init__()

        self.layer_types = config.layer_types
        self.layer_configs = config.layer_configs
        self.projection_dim = config.projection_dim
        self.final_activation = config.final_activation

        latent_dim = self.layer_configs[0].input_dim if self.layer_configs else None

        # Build the main generator body using the same modular layer system as Decoder
        self.layers = nn.ModuleList()

        # Optional linear projection from latent space to the first hidden dimension
        if self.projection_dim is not None:
            self.layers.append(nn.Linear(latent_dim, self.projection_dim))

        for layer_type, layer_config in zip(self.layer_types, self.layer_configs):
            self.layers.append(get_block(layer_type, layer_config))

        self.layers.append(get_activation(self.final_activation))

        self.network = nn.Sequential(*self.layers)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerDecoderConfig):
        super().__init__()
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.final_activation = config.final_activation

        self.layers = nn.ModuleList()

        # Optional input projection if input_dim != hidden_dim
        if self.input_dim != self.hidden_dim:
            self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))

        self.layers.append(nn.Transpose(0, 1))  # (seq_len, batch_size, hidden_dim)

        # Transformer decoder layers
        for _ in range(self.num_layers):
            self.layers.append(
                nn.TransformerDecoderLayer(
                    d_model=self.hidden_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.hidden_dim * 4,
                    dropout=self.dropout,
                )
            )

        self.layers.append(nn.Transpose(0, 1))  # Back to (batch_size, seq_len, hidden_dim)
        self.layers.append(get_activation(self.final_activation))

        self.network = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, nn.TransformerDecoderLayer):
                x = layer(x.transpose(0, 1), memory.transpose(0, 1)).transpose(0, 1)
            else:
                x = layer(x)
        return x