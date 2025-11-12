import torch
import torch.nn as nn

from ..components.composite import FeedForwardNetwork
from ..config import MLPConfig
from ..utils import get_activation

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) architecture."""
    def __init__(self, config: MLPConfig):
        super(MLP, self).__init__()
        self.network = nn.ModuleList()
        self.network.append(FeedForwardNetwork(config))
        self.network.append(get_activation(config.final_activation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)