import torch
import torch.nn as nn

from ..components.composite import FeedForwardNetwork
from ..config import CNNConfig
from ..utils import get_activation, get_block

class CNN(nn.Module):
    """A simple Convolutional Neural Network (CNN) architecture."""
    def __init__(self, config: CNNConfig):
        super(CNN, self).__init__()
        self.layer_types = config.layer_types
        self.layer_configs = config.layer_configs
        self.final_activation = config.final_activation
        self.feedforward_config = config.feedforward_config

        # Build the main generator body using the same modular layer system as Decoder
        self.layers = nn.ModuleList()

        for layer_type, layer_config in zip(self.layer_types, self.layer_configs):
            self.layers.append(get_block(layer_type, layer_config))
            
        self.layers.append(
            FeedForwardNetwork(
                self.feedforward_config
            )
        )
        self.layers.append(get_activation(self.final_activation))

        self.network = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)