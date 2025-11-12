import torch
import torch.nn as nn
from ...config import DenseLayerConfig
from ...utils import get_activation

class DenseLayer(nn.Module):
    """A fully connected dense layer with optional activation.
    
    Args:
        config (DenseLayerConfig): Configuration for the dense layer.
            - input_dim (int): Size of each input sample.
            - output_dim (int): Size of each output sample.
            - activation (str, optional): Activation function to apply. 
              Supported: 'relu', 'tanh', 'sigmoid', 'gelu'. Default is None (no activation).
    """
    def __init__(self, config: DenseLayerConfig):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(config.input_dim, config.output_dim)
        self.activation = get_activation(config.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        return x