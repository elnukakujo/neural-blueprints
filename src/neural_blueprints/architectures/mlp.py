import torch
import torch.nn as nn

from ..components.composite import FeedForwardNetwork
from ..config import MLPConfig
from ..utils import get_activation

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) architecture.
    
    Args:
        config (MLPConfig): Configuration for the MLP model.
    """
    def __init__(self, config: MLPConfig):
        super(MLP, self).__init__()
        self.config = config

        self.layers = nn.ModuleList()
        self.layers.append(FeedForwardNetwork(config))
        self.layers.append(get_activation(config.final_activation))
        self.network = nn.Sequential(*self.layers)

    def blueprint(self) -> MLPConfig:
        print(self.network)
        return self.config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            Output tensor after passing through the MLP.
        """
        return self.network(x)