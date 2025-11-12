import torch
import torch.nn as nn
from ...config import ResidualLayerConfig

class ResidualLayer(nn.Module):
    """A residual layer that adds the input to the output of a given layer.

    Args:
        layer (nn.Module): The layer to apply before adding the input.
    """
    def __init__(self, config: ResidualLayerConfig):
        super(ResidualLayer, self).__init__()
        self.layer = config.layer_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer(x)