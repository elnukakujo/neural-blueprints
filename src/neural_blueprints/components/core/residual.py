import torch
import torch.nn as nn

from ...config.components.core import ResidualLayerConfig
from ...utils import get_block

class ResidualLayer(nn.Module):
    """A residual layer that adds the input to the output of a given layer.

    Args:
        config (ResidualLayerConfig): Configuration for the residual layer.
    """
    def __init__(self, config: ResidualLayerConfig):
        super(ResidualLayer, self).__init__()

        self.layer = get_block(config.layer_type, config.layer_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual layer.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            Output tensor after applying the layer and adding the input.
        """
        return x + self.layer(x)