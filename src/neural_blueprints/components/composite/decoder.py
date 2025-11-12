import torch
import torch.nn as nn

from ...config import DecoderConfig
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