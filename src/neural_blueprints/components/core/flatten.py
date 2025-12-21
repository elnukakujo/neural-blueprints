import torch
import torch.nn as nn

from ...config.components.core import FlattenLayerConfig

class FlattenLayer(nn.Module):
    """Layer that flattens the input tensor except for the batch dimension."""

    def __init__(self, config: FlattenLayerConfig):
        super(FlattenLayer, self).__init__()
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(x)