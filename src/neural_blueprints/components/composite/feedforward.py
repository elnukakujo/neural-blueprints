import torch
from torch import nn

from .base import BaseComposite
from ..core import DenseLayer
from ...config.components.composite import FeedForwardNetworkConfig
from ...config.components.core import DenseLayerConfig

class FeedForwardNetwork(BaseComposite):
    """A feedforward neural network composed of multiple dense layers.

    Args:
        config (FeedForwardNetworkConfig): Configuration for the feedforward network.
    """
    def __init__(self, config: FeedForwardNetworkConfig):
        super(FeedForwardNetwork, self).__init__()

        input_dim = config.input_dim
        hidden_dims = config.hidden_dims
        output_dim = config.output_dim
        normalization = config.normalization
        activation = config.activation
        dropout_p = config.dropout_p
        final_activation = config.final_activation
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(DenseLayer(DenseLayerConfig(input_dim=in_dim, output_dim=[hidden_dim], normalization=normalization, activation=activation, dropout_p=dropout_p)))
            in_dim = [hidden_dim]
            
        layers.append(DenseLayer(DenseLayerConfig(input_dim=in_dim, output_dim=output_dim, normalization=None, activation=final_activation)))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feedforward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)