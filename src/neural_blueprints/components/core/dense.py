import torch
import torch.nn as nn
from ...config import DenseLayerConfig

class DenseLayer(nn.Module):
    def __init__(self, config: DenseLayerConfig):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(config.input_dim, config.output_dim)
        self.activation = self._get_activation(config.activation)

    def _get_activation(self, activation):
        if activation is None:
            return nn.Identity()
        elif activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        return x