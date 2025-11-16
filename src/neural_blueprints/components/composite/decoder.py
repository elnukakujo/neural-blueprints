import torch
import torch.nn as nn

from neural_blueprints.components.core.projection import ProjectionLayer

from ...config import DecoderConfig, TransformerDecoderConfig
from ...utils import get_block, get_activation, get_normalization

class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super(Decoder, self).__init__()

        self.layer_types = config.layer_types
        self.layer_configs = config.layer_configs
        self.projection = config.projection
        self.final_activation = config.final_activation

        # Build the main generator body using the same modular layer system as Decoder
        layers = nn.ModuleList()

        if self.projection is not None:
            layers.append(ProjectionLayer(self.projection))

        for layer_type, layer_config in zip(self.layer_types, self.layer_configs):
            layers.append(get_block(layer_type, layer_config))

        layers.append(get_activation(self.final_activation))

        self.network = nn.Sequential(*layers)
                
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
        self.final_normalization = config.final_normalization
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

        self.final_norm = get_normalization(self.final_normalization)

        # Optional normalization + activation
        self.final_act = get_activation(self.final_activation)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, tgt_seq_len, input_dim)
        memory: (batch_size, src_seq_len, hidden_dim)
        """
        x = self.input_proj(x)

        # Transpose to (seq_len, batch_size, hidden_dim)
        x = x.transpose(0, 1)
        memory = memory.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, memory)

        # Back to (batch_size, seq_len, hidden_dim)
        x = x.transpose(0, 1)
        x = self.final_norm(x)
        x = self.final_act(x)
        return x