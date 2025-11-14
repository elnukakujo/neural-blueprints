import torch
import torch.nn as nn

from ...config import ProjectionLayerConfig

class ProjectionLayer(nn.Module):
    """A simple projection layer that maps input features to a specified output dimension.

    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
    """
    def __init__(self, config: ProjectionLayerConfig):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(config.input_dim, config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)