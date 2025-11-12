import torch
import torch.nn as nn

from ...config import EncoderConfig, TransformerEncoderConfig
from ...utils import get_block, get_activation

class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super(Encoder, self).__init__()

        self.layer_types = config.layer_types
        self.layer_configs = config.layer_configs
        self.projection_dim = config.projection_dim
        self.final_activation = config.final_activation

        input_dim = self.layer_configs[0].input_dim if self.layer_configs else None

        # Build the main generator body using the same modular layer system as Decoder
        self.layers = nn.ModuleList()

        # Optional linear projection from latent space to the first hidden dimension
        if self.projection_dim is not None:
            self.layers.append(nn.Linear(input_dim, self.projection_dim))

        for layer_type, layer_config in zip(self.layer_types, self.layer_configs):
            self.layers.append(get_block(layer_type, layer_config))

        self.layers.append(get_activation(self.final_activation))

        self.network = nn.Sequential(*self.layers)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerEncoderConfig):
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

        # Transformer encoder layers
        for _ in range(self.num_layers):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.hidden_dim * 4,
                    dropout=self.dropout,
                    activation='relu'
                )
            )

        self.layers.append(nn.Transpose(0, 1))  # back to (batch_size, seq_len, hidden_dim)

        self.layers.append(get_activation(self.final_activation))

        self.network = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_dim)
        """
        return self.network(x)