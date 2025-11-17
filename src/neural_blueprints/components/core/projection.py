import torch
import torch.nn as nn

from ...config import ProjectionLayerConfig

class ProjectionLayer(nn.Module):
    """A simple projection layer that maps input features to a specified output dimension.

    Args:
        config (ProjectionLayerConfig): Configuration for the projection layer.
    """
    def __init__(self, config: ProjectionLayerConfig):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(config.input_dim, config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the projection layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.projection(x)