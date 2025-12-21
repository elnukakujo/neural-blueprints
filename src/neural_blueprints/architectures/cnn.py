import torch
import torch.nn as nn

from ..components.composite import FeedForwardNetwork
from ..config.architectures import CNNConfig
from ..utils import get_activation, get_block

class CNN(nn.Module):
    """A simple Convolutional Neural Network (CNN) architecture.
    
    Args:
        config (CNNConfig): Configuration for the CNN model.
    """
    def __init__(self, config: CNNConfig):
        super(CNN, self).__init__()
        self.config = config

        # Build the main generator body using the same modular layer system as Decoder
        self.layers = nn.ModuleList()

        for layer_type, layer_config in zip(config.layer_types, config.layer_configs):
            self.layers.append(get_block(layer_type, layer_config))
            
        self.layers.append(
            FeedForwardNetwork(
                config.feedforward_config
            )
        )
        self.layers.append(get_activation(config.final_activation))

        self.network = nn.Sequential(*self.layers)

    def blueprint(self) -> CNNConfig:
        print(self.network)
        return self.config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Output tensor after passing through the CNN.
        """
        return self.network(x)