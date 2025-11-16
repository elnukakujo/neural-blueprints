import torch
from torch import nn

from ..core import DenseLayer
from ...config import DenseLayerConfig, FeedForwardNetworkConfig

class FeedForwardNetwork(nn.Module):
    def __init__(self, config: FeedForwardNetworkConfig):
        super(FeedForwardNetwork, self).__init__()

        self.input_dim = config.input_dim
        self.hidden_dims = config.hidden_dims
        self.output_dim = config.output_dim
        self.normalization = config.normalization
        self.activation = config.activation
        
        layers = []
        in_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            config = DenseLayerConfig(input_dim=in_dim, output_dim=hidden_dim, normalization=self.normalization, activation=self.activation)
            layers.append(DenseLayer(config))
            in_dim = hidden_dim
            
        config = DenseLayerConfig(input_dim=in_dim, output_dim=self.output_dim, normalization=self.normalization, activation=None)
        layers.append(DenseLayer(config))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)