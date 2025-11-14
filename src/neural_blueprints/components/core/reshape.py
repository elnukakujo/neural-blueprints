import torch
import torch.nn as nn

from ...config import ReshapeLayerConfig

class ReshapeLayer(nn.Module):
    """A layer that reshapes its input tensor to a specified shape.
    
    Args:
        config (ReshapeLayerConfig): Configuration for the reshape layer.
            - shape (tuple of int): The target shape to reshape the input tensor to, 
              excluding the batch dimension.
    """
    def __init__(self, config: ReshapeLayerConfig):
        super(ReshapeLayer, self).__init__()
        self.target_shape = config.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, *self.target_shape)