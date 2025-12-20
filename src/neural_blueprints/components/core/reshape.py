import torch
import torch.nn as nn

from ...config.components.core import ReshapeLayerConfig

class ReshapeLayer(nn.Module):
    """A layer that reshapes its input tensor to a specified shape.
    
    Args:
        config (ReshapeLayerConfig): Configuration for the reshape layer.
    """
    def __init__(self, config: ReshapeLayerConfig):
        super(ReshapeLayer, self).__init__()
        self.target_shape = config.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the reshape layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).
        
        Returns:
            Reshaped tensor of shape (batch_size, *target_shape).
        """
        batch_size = x.size(0)
        return x.view(batch_size, *self.target_shape)